"""
Model training orchestrator for GNN and multimodal fusion models.
Complies with Project 9 Blueprint Section 8.8, 10.5 & 10.6.
"""

from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.common.config import config_manager
from src.common.logging import get_logger
from src.common.provenance import RunManifest
from src.data.schemas import ActivityRecord, Compound
from src.models.calibration import ProbabilityCalibrator
from src.models.fusion import MultimodalDTIFusionModel

logger = get_logger("train")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DTIDatasetItem:
    def __init__(self, graph: Data, protein_embedding: torch.Tensor, label: float, compound_id: str, record_id: str):
        self.graph = graph
        self.protein_embedding = protein_embedding
        self.label = label
        self.compound_id = compound_id
        self.record_id = record_id


def collate_dti_batch(batch_items: List[DTIDatasetItem]):
    graphs = [item.graph for item in batch_items]
    batch_graph = Batch.from_data_list(graphs)
    protein_embeddings = torch.stack([item.protein_embedding for item in batch_items])
    labels = torch.tensor([item.label for item in batch_items], dtype=torch.float32)
    compound_ids = [item.compound_id for item in batch_items]
    record_ids = [item.record_id for item in batch_items]
    return batch_graph, protein_embeddings, labels, compound_ids, record_ids


class ModelTrainer:
    """Orchestrates end-to-end training of Multimodal DTI Fusion Models."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        cfg = config_dict or config_manager.get_config("model_gnn")
        self.cfg = cfg
        self.checkpoints_dir = config_manager.project_root / "artifacts" / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def train_single_seed(
        self,
        train_items: List[DTIDatasetItem],
        val_items: List[DTIDatasetItem],
        seed: int = 42,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None
    ) -> Tuple[MultimodalDTIFusionModel, ProbabilityCalibrator, Dict[str, Any]]:
        """Trains one model instance with early stopping on validation split only."""
        set_seed(seed)
        t_cfg = self.cfg["training"]
        num_epochs = epochs or t_cfg.get("epochs", 35)
        bs = batch_size or t_cfg.get("batch_size", 32)
        learning_rate = lr or t_cfg.get("learning_rate", 0.001)
        patience = t_cfg.get("early_stopping_patience", 7)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_in_dim = train_items[0].graph.x.size(-1) if train_items else self.cfg["gnn_encoder"].get("node_in_dim", 43)
        model = MultimodalDTIFusionModel(
            node_in_dim=node_in_dim,
            gnn_hidden_dim=self.cfg["gnn_encoder"]["output_dim"],
            protein_emb_dim=self.cfg["protein_projection"]["input_dim"],
            projection_dim=self.cfg["protein_projection"]["output_dim"],
            dropout=self.cfg["gnn_encoder"]["dropout"]
        ).to(device)

        # Imbalance weighting (computed only on training split!)
        train_labels = [item.label for item in train_items]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor([max(1.0, n_neg / max(1.0, n_pos))], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=t_cfg.get("weight_decay", 1e-4))

        best_val_loss = float("inf")
        best_val_state = None
        patience_counter = 0

        logger.info(f"Starting model training (seed={seed}, N_train={len(train_items)}, N_val={len(val_items)}, pos_weight={pos_weight.item():.2f})...")

        for epoch in range(1, num_epochs + 1):
            # Training
            model.train()
            random.shuffle(train_items)
            train_losses = []

            for i in range(0, len(train_items), bs):
                batch_slice = train_items[i:i + bs]
                if not batch_slice:
                    continue
                bg, pe, y, _, _ = collate_dti_batch(batch_slice)
                bg = bg.to(device)
                pe = pe.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits, _, _ = model(bg.x, bg.edge_index, pe, bg.batch)
                loss = criterion(logits, y.view(-1))
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            val_logits_list = []
            val_targets_list = []

            with torch.no_grad():
                for i in range(0, len(val_items), bs):
                    batch_slice = val_items[i:i + bs]
                    if not batch_slice:
                        continue
                    bg, pe, y, _, _ = collate_dti_batch(batch_slice)
                    bg = bg.to(device)
                    pe = pe.to(device)
                    y = y.to(device)

                    logits, _, _ = model(bg.x, bg.edge_index, pe, bg.batch)
                    v_loss = criterion(logits, y.view(-1))
                    val_losses.append(v_loss.item())
                    val_logits_list.extend(logits.cpu().numpy())
                    val_targets_list.extend(y.cpu().numpy())

            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter == patience:
                logger.debug(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

        # Restore best model checkpoint
        if best_val_state:
            model.load_state_dict(best_val_state)

        # Fit probability calibrator on validation predictions
        model.eval()
        final_val_logits = []
        final_val_targets = []
        with torch.no_grad():
            for i in range(0, len(val_items), bs):
                batch_slice = val_items[i:i + bs]
                bg, pe, y, _, _ = collate_dti_batch(batch_slice)
                logits, _, _ = model(bg.to(device).x, bg.to(device).edge_index, pe.to(device), bg.to(device).batch)
                final_val_logits.extend(logits.cpu().numpy())
                final_val_targets.extend(y.cpu().numpy())

        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(np.array(final_val_logits), np.array(final_val_targets))

        training_summary = {
            "seed": seed,
            "best_val_loss": round(best_val_loss, 4),
            "epochs_run": epoch,
            "pos_weight": round(pos_weight.item(), 4),
            "train_samples": len(train_items),
            "val_samples": len(val_items),
        }
        return model, calibrator, training_summary

    def train_deep_ensemble(
        self,
        train_items: List[DTIDatasetItem],
        val_items: List[DTIDatasetItem],
        seeds: Optional[List[int]] = None
    ) -> Tuple[List[MultimodalDTIFusionModel], ProbabilityCalibrator, Dict[str, Any]]:
        """Trains deep ensemble across multiple seeds."""
        ensemble_seeds = seeds or self.cfg["uncertainty"].get("ensemble_seeds", [42, 101, 777])
        models = []
        primary_calibrator = None
        summaries = []

        for s in ensemble_seeds:
            model, cal, summary = self.train_single_seed(train_items, val_items, seed=s)
            models.append(model)
            summaries.append(summary)
            if primary_calibrator is None:
                primary_calibrator = cal

        # Save ensemble checkpoints
        checkpoint_paths = []
        timestamp = int(time.time())
        for idx, (m, s) in enumerate(zip(models, ensemble_seeds)):
            ckpt_path = self.checkpoints_dir / f"dti_model_seed_{s}_{timestamp}.pt"
            torch.save(m.state_dict(), ckpt_path)
            checkpoint_paths.append(str(ckpt_path))

        report = {
            "ensemble_size": len(models),
            "seeds": ensemble_seeds,
            "summaries": summaries,
            "checkpoint_paths": checkpoint_paths,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return models, primary_calibrator, report
