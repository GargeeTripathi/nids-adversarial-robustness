"""
The four baseline NIDS architectures compared in RO1.

All four consume the same input - a flat vector of encoded flow features - and produce two
logits (benign / attack). Keeping the interface identical is what makes the robustness
comparison fair: any difference in adversarial vulnerability is attributable to the
architecture rather than to differences in input representation or output head.

Adapting sequence/graph models to tabular flow records
------------------------------------------------------
CNNs, LSTMs and GNNs expect structured input (grids, sequences, graphs) that a flow feature
vector does not natively have. Published NIDS work handles this in varied and often
under-documented ways, which is a real obstacle to reproducibility. The approach taken here
is stated explicitly for each model, and the *limitation* is stated too:

* **CNN** - treats the feature vector as a 1-D signal and slides kernels over it. This
  presupposes that adjacent features are related, which for a flow record is only loosely
  true (the canonical ordering does group related quantities). The convolution therefore
  acts as a local feature-interaction detector rather than a spatial one.

* **LSTM** - reads the feature vector as a pseudo-sequence, one feature per timestep. This
  is *not* temporal modelling: a single flow record has no time axis. True temporal NIDS
  would sequence multiple flows per host. This is the standard "LSTM on tabular data"
  baseline found in the literature, and its main value here is as a recurrent-architecture
  datapoint for the robustness comparison, not as a principled temporal model.

* **Transformer** - treats each feature as a token with a learned embedding, letting
  self-attention model arbitrary pairwise feature interactions. Unlike the CNN this makes
  no adjacency assumption, which is a better fit for tabular data (cf. TabTransformer/FT-
  Transformer).

* **GNN** - needs a graph. NSL-KDD contains no host identifiers, so a host-communication
  graph cannot be built from it. A k-nearest-neighbour similarity graph over flow feature
  vectors is used instead, and message passing aggregates information from each flow's
  neighbours. This is a genuine architectural difference (relational inductive bias) but it
  is NOT the same thing as a topology-aware NIDS such as E-GraphSAGE, and results should
  not be read as evidence about those.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 2


# ======================================================================================
# CNN
# ======================================================================================


class CNNNIDS(nn.Module):
    """1-D convolutional network over the feature vector."""

    # Channel widths kept modest (32/64/64 rather than 64/128/128): the wider variant cost
    # ~4x the compute for no measurable accuracy gain on 123 input features, and the study
    # retrains every model dozens of times across the poisoning sweep.
    def __init__(self, n_features: int, channels: tuple[int, ...] = (32, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Global pooling rather than flatten: keeps the head's parameter count independent
        # of n_features, so the same hyper-parameters work across datasets with different
        # feature counts (needed for cross-domain runs).
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                    # (B, F) -> (B, 1, F)
        return self.head(self.pool(self.features(x)))


# ======================================================================================
# LSTM
# ======================================================================================


class LSTMNIDS(nn.Module):
    """
    Bidirectional LSTM reading the feature vector as a chunked pseudo-sequence.

    Features are grouped into blocks of `chunk_size` and each block becomes one timestep,
    rather than the naive one-scalar-per-timestep formulation. Two reasons:

    * **Cost.** Recurrence cannot be parallelised across timesteps, so a 123-step sequence
      is ~8x the sequential work of a 16-step one. On CPU the naive version was the single
      bottleneck of the whole study.
    * **Representation.** A single scalar carries almost no information for a recurrent
      cell to gate on; a block of adjacent features (which, in the canonical ordering, are
      related quantities) gives each step something substantive to read.

    This remains a pseudo-sequence: a flow record has no genuine time axis. See the module
    docstring for why that caveat matters when interpreting the LSTM's robustness numbers.
    """

    def __init__(self, n_features: int, hidden_size: int = 96, num_layers: int = 1,
                 chunk_size: int = 8, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.chunk_size = chunk_size
        # Pad up to a whole number of chunks; padding is zeros, which the LSTM learns to
        # ignore since the padded positions are constant across every sample.
        self.padded_size = ((n_features + chunk_size - 1) // chunk_size) * chunk_size
        self.n_steps = self.padded_size // chunk_size

        self.lstm = nn.LSTM(
            input_size=chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),   # *2 for the two directions
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padded_size > self.n_features:
            x = F.pad(x, (0, self.padded_size - self.n_features))
        x = x.view(x.shape[0], self.n_steps, self.chunk_size)
        output, _ = self.lstm(x)
        return self.head(output.mean(dim=1))  # mean-pool over steps


# ======================================================================================
# Transformer
# ======================================================================================


class TransformerNIDS(nn.Module):
    """
    Feature-tokenising Transformer encoder.

    Each scalar feature is projected to a d_model embedding and given a learned positional
    embedding identifying *which* feature it is; self-attention then models interactions
    between arbitrary feature pairs. A CLS token aggregates for classification.
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, chunk_size: int = 8, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.chunk_size = chunk_size

        # Features are grouped into chunks, each becoming one token, rather than one token
        # per scalar feature. Self-attention is O(n_tokens^2), so 124 single-feature tokens
        # cost ~60x more than 16 chunked ones - on CPU that was 4.0 s/batch versus ~0.07,
        # the difference between a feasible study and an infeasible one. Grouping adjacent
        # features is also reasonable modelling: the canonical ordering places related
        # quantities together, so a chunk is a coherent sub-record rather than an arbitrary
        # slice. Attention still models all-pairs interactions, now between feature groups.
        self.padded_size = ((n_features + chunk_size - 1) // chunk_size) * chunk_size
        self.n_tokens = self.padded_size // chunk_size

        self.value_projection = nn.Linear(chunk_size, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(self.n_tokens, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm: markedly more stable at this depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.padded_size > self.n_features:
            x = F.pad(x, (0, self.padded_size - self.n_features))
        x = x.view(batch, self.n_tokens, self.chunk_size)

        tokens = self.value_projection(x)                      # (B, T, d)
        tokens = tokens + self.feature_embedding.unsqueeze(0)  # identify each group
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)               # (B, 1+T, d)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])                        # classify from CLS


# ======================================================================================
# GNN
# ======================================================================================


class GraphSAGELayer(nn.Module):
    """
    One GraphSAGE-style message-passing layer.

    Implemented directly rather than via torch-geometric to keep the dependency footprint
    small and the mechanics visible: `h_i' = W_self · h_i + W_neigh · mean_{j∈N(i)} h_j`.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.neighbour_transform = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, in_dim) node features.
            adjacency: (N, N) row-normalised adjacency; row i averages i's neighbours.
        """
        neighbour_mean = adjacency @ h
        return self.self_transform(h) + self.neighbour_transform(neighbour_mean)


class GNNNIDS(nn.Module):
    """
    GraphSAGE over a k-NN similarity graph of flow records.

    The graph is built at inference time from the batch itself (see `build_knn_adjacency`),
    which keeps the model usable in the same `forward(x) -> logits` interface as the others
    and, importantly, keeps it differentiable end-to-end with respect to `x` so that
    gradient-based attacks apply identically.

    Caveat: because the graph is rebuilt per batch, an adversarial perturbation changes not
    only a node's own features but also its neighbourhood. That is realistic - an attacker
    altering traffic does change its similarity to other traffic - but it makes the attack
    surface subtly larger than for the other three models, and is discussed in the results.
    """

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 2,
                 k: int = 10, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.k = k

        self.input_projection = nn.Linear(n_features, hidden_dim)
        self.layers = nn.ModuleList(
            GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, NUM_CLASSES)

    def build_knn_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Row-normalised k-NN adjacency from pairwise Euclidean distance."""
        n = x.shape[0]
        k = min(self.k, max(n - 1, 1))

        distances = torch.cdist(x, x)
        distances = distances + torch.eye(n, device=x.device) * 1e9   # exclude self
        _, neighbour_idx = distances.topk(k, dim=1, largest=False)

        adjacency = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        adjacency.scatter_(1, neighbour_idx, 1.0 / k)
        return adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # detach() on the adjacency: gradients flow through node *features*, not through
        # the discrete top-k neighbour selection, which has zero gradient almost everywhere
        # anyway. Without this the attack gradient is dominated by meaningless noise from
        # the distance computation.
        adjacency = self.build_knn_adjacency(x).detach()

        h = self.input_projection(x)
        for layer, norm in zip(self.layers, self.norms):
            h = self.dropout(F.relu(norm(layer(h, adjacency))))
        return self.head(h)


# ======================================================================================
# Registry
# ======================================================================================

ARCHITECTURES = {
    "cnn": CNNNIDS,
    "lstm": LSTMNIDS,
    "transformer": TransformerNIDS,
    "gnn": GNNNIDS,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    """Instantiate an architecture by name."""
    if name not in ARCHITECTURES:
        raise KeyError(f"Unknown architecture '{name}'. Available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](n_features=n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
