import math

import torch
from torch import nn

from nets.graph_encoder import GraphAttentionEncoder


class EchoTaskEncoder(nn.Module):
    """ECHO-style dual-modality task encoder for CHRSP.

    The encoder keeps the AM task-attribute stream and adds an edge stream
    derived from pairwise Manhattan distances. Edge-aware task features are
    fused into task embeddings through gated cross-attention.
    """

    def __init__(
            self,
            node_dim,
            embed_dim,
            n_heads,
            n_layers,
            normalization='batch',
            edge_dim=None
    ):
        super(EchoTaskEncoder, self).__init__()

        edge_dim = embed_dim if edge_dim is None else edge_dim
        assert edge_dim == embed_dim, "edge_dim must match embed_dim for gated fusion"

        self.embed_dim = embed_dim
        self.node_init = nn.Linear(node_dim, embed_dim, bias=False)
        self.edge_key = nn.Linear(1, embed_dim, bias=False)
        self.edge_value = nn.Linear(1, embed_dim, bias=False)
        self.edge_score = nn.Linear(embed_dim, 1, bias=False)
        self.edge_to_task = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.fusion_gate = nn.Linear(2 * embed_dim, embed_dim)
        self.task_encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            normalization=normalization
        )

    def forward(self, node_features, coords):
        task_embeddings = self.node_init(node_features)

        pairwise_dist = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=1, dim=-1)
        edge_inputs = pairwise_dist.unsqueeze(-1)
        edge_k = self.edge_key(edge_inputs)
        edge_v = self.edge_value(edge_inputs)

        edge_scores = self.edge_score(edge_k).squeeze(-1) / math.sqrt(self.embed_dim)
        edge_weights = torch.softmax(edge_scores, dim=-1)
        edge_context = torch.sum(edge_weights.unsqueeze(-1) * edge_v, dim=2)

        edge_attended, _ = self.edge_to_task(task_embeddings, edge_context, edge_context)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((task_embeddings, edge_attended), dim=-1)))
        fused_embeddings = task_embeddings + gate * edge_attended

        return self.task_encoder(fused_embeddings)

