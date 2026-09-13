import torch
from torch import nn
import math



class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    # Forward the optional mask to the attention layer.
    def forward(self, x, y=None, mask=None):
        if y is not None:
            # 支持双输入（交叉注意力）并传递mask
            return x + self.module(x, y, mask=mask)
        # 单输入兼容（自注意力或FFN）
        # 注意: 只有注意力模块会用到mask，FFN会自动忽略它
        return x + self.module(x, mask=mask)



class MultiHeadAttention(nn.Module):
    # ... (init方法和init_parameters方法保持不变) ...
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # parameter表示需要在训练中进行优化的参数
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # Accept an optional task mask.
    def forward(self, q, h, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim) 键和值
        :param mask: mask (batch_size, graph_size) or viewable as that.
                     True表示该位置需要被mask掉 (即任务已完成)
        """
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        # ... (前面的计算保持不变) ...
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Broadcast the task mask over attention heads and query positions.
        if mask is not None:
            # 1. 鲁棒地将mask调整为2D: (batch_size, graph_size)
            #    这可以处理 (B, S), (B, 1, S) 等多种输入形状
            reshaped_mask = mask.view(batch_size, graph_size)

            # 2. 为广播做准备：(B, S) -> (1, B, 1, S)
            #    以匹配compatibility的形状 (n_heads, batch_size, n_query, graph_size)
            broadcast_mask = reshaped_mask.unsqueeze(0).unsqueeze(2)

            # 3. 应用广播后的mask
            compatibility[broadcast_mask.expand_as(compatibility)] = -math.inf

        attn = torch.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out




class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm,
        }.get(normalization, None)
        if normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim)
        else:
            self.normalizer = normalizer_class(embed_dim, affine=True)


        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):
    """
    一个完整的Transformer编码器层，包含多头注意力、残差连接、层归一化和前馈网络。
    这个版本不再使用有歧义的SkipConnection，而是手动实现残差连接，代码更清晰。
    """

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        # 1. 注意力模块 (不再用SkipConnection包装)
        self.attention = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        # 2. 前馈网络模块 (不再用SkipConnection包装)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )

        # 3. 两个层归一化模块
        self.norm1 = Normalization(embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim, normalization=normalization)

    def forward(self, x, y, mask=None):
        """
        前向传播。
        Args:
            x (Tensor): Query 张量
            y (Tensor): Key/Value 张量 (在交叉注意力中使用)
            mask (Tensor, optional): 应用于Key/Value的掩码. Defaults to None.
        """
        # --- 第一部分: 多头注意力 + 残差 + 归一化 ---
        # x 是 query (agent_embedding), y 是 key/value (task_embedding)
        # 1. 计算注意力输出
        attn_output = self.attention(x, y, mask=mask)
        # 2. 手动实现残差连接, 然后进行归一化
        h = self.norm1(x + attn_output)

        # --- 第二部分: 前馈网络 + 残差 + 归一化 ---
        # 1. 计算前馈网络输出
        ffn_output = self.ffn(h)
        # 2. 手动实现残差连接, 然后进行归一化
        out = self.norm2(h + ffn_output)

        return out


class AgentLearnTaskCrossAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            normalization='layer',
            feed_forward_hidden=512,
    ):
        super(AgentLearnTaskCrossAttention, self).__init__()

        self.task_robot_attention = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden,
                                                            normalization)

    # Pass the mask through the encoder layers.
    def forward(self, agent_embedding, task_embedding, mask):
        """
        Args:
            agent_embedding (Tensor): (batch, num_agents, dim) -> (1024, 12, 128)
            task_embedding (Tensor): (batch, num_tasks, dim) -> (1024, 20, 128)
            mask (Tensor): (batch, num_tasks) -> (1024, 20)
                         True表示任务已完成，需要被mask掉。
        """
        # 调用MHA层，传入mask
        contextualized_agent_embedding = self.task_robot_attention(agent_embedding, task_embedding, mask=mask)

        return (
            contextualized_agent_embedding,  # (batch_size, num_agents, embed_dim)
            contextualized_agent_embedding.mean(dim=1),  # (batch_size, embed_dim)
        )
