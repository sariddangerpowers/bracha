import torch
import torch.nn as nn
import numpy as np
from network_transformer import TransformerFlashback
from scipy.sparse import csr_matrix


class TransformerTrainer():
    """Instantiates TransformerFlashback module with spatial and temporal weight functions.
    Performs loss computation and prediction. No hidden state management needed.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight,
                 transition_graph, spatial_graph, friend_graph, use_graph_user,
                 use_spatial_graph, interact_graph):
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph

    def __str__(self):
        return 'Use TransformerFlashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count

    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, d_model, n_heads, n_layers, dropout, device):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))

        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))

        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = TransformerFlashback(
            loc_count, user_count, d_model, n_heads, n_layers, dropout,
            f_t, f_s, self.lambda_loc, self.lambda_user, self.use_weight,
            self.graph, self.spatial_graph, self.friend_graph,
            self.use_graph_user, self.use_spatial_graph, self.interact_graph
        ).to(device)

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, active_users):
        """Predict next locations. Returns logits transposed to (batch, seq, loc_count)."""
        self.model.eval()
        out = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, active_users)
        out_t = out.transpose(0, 1)
        return out_t

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, active_users):
        """Compute training loss (cross-entropy over all sequence positions)."""
        self.model.train()
        out = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l
