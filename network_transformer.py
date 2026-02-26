import torch
import torch.nn as nn
import numpy as np
from utils import sparse_matrix_to_tensor, calculate_random_walk_matrix
from scipy.sparse import identity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerFlashback(nn.Module):
    """Transformer-based Flashback: Replaces the RNN + hand-crafted temporal-spatial
    attention with a causal Transformer whose attention incorporates temporal-spatial
    decay biases. The GCN graph processing is kept identical to the original.
    """

    def __init__(self, input_size, user_count, d_model, n_heads, n_layers, dropout,
                 f_t, f_s, lambda_loc, lambda_user, use_weight,
                 graph, spatial_graph, friend_graph, use_graph_user,
                 use_spatial_graph, interact_graph):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.d_model = d_model
        self.n_heads = n_heads
        self.f_t = f_t
        self.f_s = f_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph

        # --- Graph setup (identical to original Flashback) ---
        self.I = identity(graph.shape[0], format='coo')
        self.graph = sparse_matrix_to_tensor(
            calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))

        self.spatial_graph = spatial_graph
        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(
                calculate_random_walk_matrix(interact_graph))
        else:
            self.interact_graph = None

        # --- Embeddings ---
        self.encoder = nn.Embedding(input_size, d_model)       # location embedding
        self.user_encoder = nn.Embedding(user_count, d_model)   # user embedding
        self.time_encoder = nn.Embedding(168, d_model)          # time slot embedding (24h * 7days)
        self.coord_encoder = nn.Linear(2, d_model)              # GPS coordinate projection

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False  # input shape: (seq, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output ---
        self.fc = nn.Linear(2 * d_model, input_size)

        # --- Learnable scaling for temporal-spatial attention bias ---
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _compute_attention_bias(self, t, s, user_loc_similarity, seq_len, user_len):
        """Compute temporal-spatial attention bias matrix.

        Returns a combined causal mask + bias of shape (user_len * n_heads, seq_len, seq_len).
        Positions where j > i (future) are -inf (causal). For j <= i, the bias is:
            alpha * f_t(t_i - t_j) * f_s(||s_i - s_j||) * user_loc_sim_j
        """
        # Pairwise temporal distances: (seq, seq, batch)
        # t shape: (seq, batch)
        t_i = t.unsqueeze(1)  # (seq, 1, batch)
        t_j = t.unsqueeze(0)  # (1, seq, batch)
        dist_t = t_i - t_j    # (seq, seq, batch)

        # Pairwise spatial distances: (seq, seq, batch)
        # s shape: (seq, batch, 2)
        s_i = s.unsqueeze(1)  # (seq, 1, batch, 2)
        s_j = s.unsqueeze(0)  # (1, seq, batch, 2)
        dist_s = torch.norm(s_i - s_j, dim=-1)  # (seq, seq, batch)

        # Apply decay functions
        # f_t and f_s expect (val, user_len) but we pass full matrices
        # f_t(delta_t) = ((cos(delta_t * 2pi / 86400) + 1) / 2) * exp(-(delta_t / 86400 * lambda_t))
        # f_s(delta_s) = exp(-(delta_s * lambda_s))
        temporal_weight = self.f_t(dist_t, user_len)  # (seq, seq, batch)
        spatial_weight = self.f_s(dist_s, user_len)    # (seq, seq, batch)

        # Combined bias: (seq, seq, batch)
        bias = self.alpha * temporal_weight * spatial_weight

        # Incorporate user-location similarity as a key-side bias
        # user_loc_similarity shape: (user_len, seq_len) — similarity at each source position j
        # Expand: bias[i, j, b] *= user_loc_sim[b, j]
        user_loc_sim_expanded = user_loc_similarity.unsqueeze(1)  # (batch, 1, seq)  -- broadcasts over i
        bias = bias.permute(2, 0, 1)  # (batch, seq_i, seq_j)
        bias = bias * user_loc_sim_expanded

        # Causal mask: positions where j > i get -inf
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=t.device), diagonal=1).bool()
        bias = bias.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Expand to (batch * n_heads, seq, seq)
        bias = bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (batch, n_heads, seq, seq)
        bias = bias.reshape(user_len * self.n_heads, seq_len, seq_len)

        return bias

    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, active_user):
        seq_len, user_len = x.size()

        # --- 1. GCN-refined location embeddings (same as original) ---
        p_u = self.user_encoder(active_user)  # (1, user_len, d_model)
        p_u = p_u.view(user_len, self.d_model)

        graph = self.graph.to(x.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(x.device)

        if self.use_spatial_graph:
            spatial_graph = (self.spatial_graph *
                             self.lambda_loc + self.I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(spatial_graph).to(x.device)
            encoder_weight += torch.sparse.mm(spatial_graph, loc_emb).to(x.device)
            encoder_weight /= 2

        new_x_emb = []
        for i in range(seq_len):
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)
        x_emb = torch.stack(new_x_emb, dim=0)  # (seq, batch, d_model)

        # --- 2. User-POI interaction (same as original) ---
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size))).to(x.device))
        encoder_weight_for_user = loc_emb
        interact_graph = self.interact_graph.to(x.device)
        encoder_weight_user = torch.sparse.mm(
            interact_graph, encoder_weight_for_user).to(x.device)

        user_preference = torch.index_select(
            encoder_weight_user, 0, active_user.squeeze(0)).unsqueeze(0)
        user_loc_similarity = torch.exp(
            -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(x.device)
        user_loc_similarity = user_loc_similarity.permute(1, 0)  # (user_len, seq_len)

        # --- 3. Input enrichment (NEW) ---
        t_emb = self.time_encoder(t_slot)     # (seq, batch, d_model)
        s_emb = self.coord_encoder(s.float()) # (seq, batch, d_model)
        src = x_emb + t_emb + s_emb           # additive combination

        # --- 4. Temporal-spatial attention bias (NEW — replaces Flashback loop) ---
        attn_bias = self._compute_attention_bias(t, s, user_loc_similarity, seq_len, user_len)

        # --- 5. Transformer forward ---
        out = self.transformer_encoder(src, mask=attn_bias)  # (seq, batch, d_model)

        # --- 6. Output projection (same structure as original) ---
        out_pu = torch.zeros(seq_len, user_len, 2 * self.d_model, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out[i], p_u], dim=1)

        y_linear = self.fc(out_pu)  # (seq, batch, input_size)

        return y_linear
