import torch
from torch import nn
import torch.nn.functional as F


class TopKMoEBase(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super(TopKMoEBase, self).__init__()
        assert num_experts > 0
        assert 0 < top_k <= num_experts
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.moe_aux_loss = torch.tensor(0.0)

    def _route(self, x, expert_outputs):
        original_shape = x.size()[:-1]
        x_flat = x.contiguous().view(-1, self.input_dim)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_weight = topk_prob / topk_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        stacked = torch.stack(expert_outputs, dim=1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, stacked.size(-1))
        selected = stacked.gather(1, gather_idx)
        out = (selected * topk_weight.unsqueeze(-1)).sum(dim=1)

        selected_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float().sum(dim=1)
        load = selected_mask.mean(dim=0) / float(self.top_k)
        importance = probs.mean(dim=0)
        self.moe_aux_loss = self.num_experts * torch.sum(importance * load)
        return out.view(*original_shape, -1)


class TopKMoEFeedForward(TopKMoEBase):
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_experts=4, top_k=2):
        super(TopKMoEFeedForward, self).__init__(input_dim, num_experts, top_k)
        output_dim = input_dim if output_dim is None else output_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        x_flat = x.contiguous().view(-1, self.input_dim)
        expert_outputs = [expert(x_flat) for expert in self.experts]
        return self._route(x, expert_outputs)


class TopKMoELinear(TopKMoEBase):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2, bias=False):
        super(TopKMoELinear, self).__init__(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=bias)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        x_flat = x.contiguous().view(-1, self.input_dim)
        expert_outputs = [expert(x_flat) for expert in self.experts]
        return self._route(x, expert_outputs)

