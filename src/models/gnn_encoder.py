"""
Molecular Graph Neural Network (GNN) encoder.
Complies with Project 9 Blueprint Section 8.6, 10.2 & 10.4.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class MolecularGNNEncoder(nn.Module):
    """Encodes molecular graphs into dense compound embedding vectors."""

    def __init__(
        self,
        node_in_dim: int = 43,
        hidden_dims: Tuple[int, ...] = (64, 64, 64),
        output_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Conv layers
        self.conv1 = GCNConv(node_in_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])

        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])

        # Readout projection (mean + max pooling = 2 * hidden_dims[-1] -> output_dim)
        self.fc_pool = nn.Linear(hidden_dims[-1] * 2, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            graph_embedding: [batch_size, output_dim]
            node_embeddings: [num_total_nodes, hidden_dims[-1]] (used for atom-level attribution)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        node_embeddings = h

        # Readout (global mean + global max pooling)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max], dim=-1)

        graph_embedding = F.relu(self.fc_pool(h_combined))
        graph_embedding = F.dropout(graph_embedding, p=self.dropout_rate, training=self.training)

        return graph_embedding, node_embeddings


class MoleculeOnlyGNN(nn.Module):
    """Standalone GNN classifier without protein representation (baseline)."""

    def __init__(self, node_in_dim: int = 43, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = MolecularGNNEncoder(
            node_in_dim=node_in_dim,
            hidden_dims=(hidden_dim, hidden_dim, hidden_dim),
            output_dim=hidden_dim,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        graph_emb, _ = self.encoder(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits.squeeze(-1)
