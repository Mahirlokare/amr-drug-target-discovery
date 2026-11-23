"""
Multimodal Drug-Target Interaction (GNN + ESM-2) Fusion Model.
Complies with Project 9 Blueprint Section 10.4 (Fusion model).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_encoder import MolecularGNNEncoder


class MultimodalDTIFusionModel(nn.Module):
    """
    Combines molecular graph representations (GNN) and protein language model
    sequence embeddings (ESM-2) into unified drug-target interaction predictions.
    """

    def __init__(
        self,
        node_in_dim: int = 43,
        gnn_hidden_dim: int = 64,
        protein_emb_dim: int = 320,
        projection_dim: int = 64,
        fusion_hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2
    ):
        super().__init__()
        # Molecular Graph Encoder
        self.gnn_encoder = MolecularGNNEncoder(
            node_in_dim=node_in_dim,
            hidden_dims=(gnn_hidden_dim, gnn_hidden_dim, gnn_hidden_dim),
            output_dim=projection_dim,
            dropout=dropout
        )

        # Protein Sequence ESM-2 Projection Network
        self.protein_projector = nn.Sequential(
            nn.Linear(protein_emb_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # Multimodal Interaction & Fusion Head
        # Concatenation of molecule vector (64-d) + protein vector (64-d) = 128-d
        combined_dim = projection_dim * 2

        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dims[0]),
            nn.LayerNorm(fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dims[0], fusion_hidden_dims[1]),
            nn.LayerNorm(fusion_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dims[1], fusion_hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dims[2], 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        protein_embedding: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: atom features [N, node_in_dim]
            edge_index: graph edges [2, E]
            protein_embedding: ESM-2 sequence embedding [batch_size, protein_emb_dim] or [protein_emb_dim]
            batch: PyG batch indicator [N]
        Returns:
            logit: [batch_size]
            mol_vec: [batch_size, projection_dim]
            prot_vec: [batch_size, projection_dim]
        """
        mol_vec, _ = self.gnn_encoder(x, edge_index, batch)

        # Handle 1D protein embedding passed for single batch
        if protein_embedding.dim() == 1:
            # Broadcast across batch
            batch_size = mol_vec.size(0)
            protein_embedding = protein_embedding.unsqueeze(0).expand(batch_size, -1)
        elif protein_embedding.size(0) != mol_vec.size(0):
            # Broadcast if single target embedding for multiple candidate molecules
            protein_embedding = protein_embedding.expand(mol_vec.size(0), -1)

        prot_vec = self.protein_projector(protein_embedding)

        # Concatenate multimodal representations
        fused = torch.cat([mol_vec, prot_vec], dim=-1)
        logit = self.fusion_head(fused).view(-1)

        return logit, mol_vec, prot_vec
