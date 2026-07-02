import torch
from torch import nn
import math



class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x, y=None, mask=None):
        if y is not None:
            return x + self.module(x, y, mask=mask)
        return x + self.module(x, mask=mask)



class MultiHeadAttention(nn.Module):
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

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim) 
        :param mask: mask (batch_size, graph_size) or viewable as that.
                     Truemask ()
        """
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            reshaped_mask = mask.view(batch_size, graph_size)

            broadcast_mask = reshaped_mask.unsqueeze(0).unsqueeze(2)

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
    Transformer
    SkipConnection
    """

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.attention = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )

        self.norm1 = Normalization(embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim, normalization=normalization)

    def forward(self, x, y, mask=None):
        """
        
        Args:
            x (Tensor): Query 
            y (Tensor): Key/Value  ()
            mask (Tensor, optional): Key/Value. Defaults to None.
        """
        attn_output = self.attention(x, y, mask=mask)
        h = self.norm1(x + attn_output)

        ffn_output = self.ffn(h)
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

    def forward(self, agent_embedding, task_embedding, mask):
        """
        Args:
            agent_embedding (Tensor): (batch, num_agents, dim) -> (1024, 12, 128)
            task_embedding (Tensor): (batch, num_tasks, dim) -> (1024, 20, 128)
            mask (Tensor): (batch, num_tasks) -> (1024, 20)
                         Truemask
        """
        contextualized_agent_embedding = self.task_robot_attention(agent_embedding, task_embedding, mask=mask)

        return (
            contextualized_agent_embedding,  # (batch_size, num_agents, embed_dim)
            contextualized_agent_embedding.mean(dim=1),  # (batch_size, embed_dim)
        )

