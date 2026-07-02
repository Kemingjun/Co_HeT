import torch
import torch.nn as nn

class EnvGRUUpdater(nn.Module):
    """
     token  GRU  1  GRU
    """
    def __init__(self, input_dim, hidden_dim):
        super(EnvGRUUpdater, self).__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, context, prev_token):
        """
        context: (batch_size, input_dim)
        prev_token: (batch_size, hidden_dim)
        """
        new_token = self.gru(context, prev_token)
        return new_token


class VehicleGRUUpdater(nn.Module):
    """
     GRUm  GRU
    """
    def __init__(self, num_vehicles, input_dim, hidden_dim):
        super(VehicleGRUUpdater, self).__init__()
        self.num_vehicles = num_vehicles
        self.grus = nn.ModuleList([
            nn.GRUCell(input_dim, hidden_dim) for _ in range(num_vehicles)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(num_vehicles * 3, input_dim, bias=False) for _ in range(num_vehicles)
        ])

    def forward(self, prev_tokens_tensor, env_context):
        """
        Args:
            context_tensor: Tensor, shape = [batch_size, num_vehicles, input_dim]
            prev_tokens_tensor: Tensor, shape = [batch_size, num_vehicles, hidden_dim]

        Returns:
            new_tokens_tensor: Tensor, shape = [batch_size, num_vehicles, hidden_dim]
        """
        batch_size = env_context.size(0)
        hidden_dim = prev_tokens_tensor.size(-1)

        new_tokens_tensor = torch.zeros(batch_size, self.num_vehicles, hidden_dim, device=env_context.device)

        for i in range(self.num_vehicles):
            # context_i = context_tensor[:, i, :]  # [batch_size, input_dim]
            prev_token_i = prev_tokens_tensor[:, i, :]  # [batch_size, hidden_dim]
            # context = torch.cat([context_i, env_context], dim=-1)
            context = self.linears[i](env_context)
            new_token_i = self.grus[i](context, prev_token_i)  # [batch_size, hidden_dim]

            new_tokens_tensor[:, i, :] = new_token_i

        return new_tokens_tensor

