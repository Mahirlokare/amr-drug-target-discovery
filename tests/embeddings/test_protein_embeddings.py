"""
Protein embedding and language model service test suite.
Verifies sequence hashing, 320-dim tensor shapes, caching behavior,
and transparent provenance metadata.
Complies with Section 4 & 14.
"""

import pytest
import torch
import json

from src.features.protein_esm import ESMEmbeddingService


@pytest.fixture
def esm_service(tmp_path):
    return ESMEmbeddingService(cache_dir=tmp_path)


def test_sequence_hashing(esm_service):
    seq1 = "maflngqlvtgaaqgigrayalalakagaqvvvndlt"
    seq2 = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLT"
    assert esm_service._sequence_hash(seq1) == esm_service._sequence_hash(seq2)


def test_embedding_dimension_and_norm(esm_service):
    seq = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLTDAKVAAEVAAAGANAVALHCDVTDESQISALVAAAVERYGTVDIVVSNAGIAVPADGVEHDVTSDGWETMLNVNAAGVFLGLQAAIPLLAKGGSVVINMSSVSGQMGYPALGAYSASKAAVNGLVRNLALEHAPVRVNCINPGLVRTRLQERLLADPYQAVQAGWAPETLDKIAALAGVDTSGYYSLTPRVLRVTPGDAEYVGKPVTVIQPGGFTQVDAYD"
    emb, seq_hash = esm_service.get_embedding(seq, target_id="TARGET-001")

    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (320,)
    assert emb.dtype == torch.float32
    norm = float(torch.norm(emb))
    assert abs(norm - 1.0) < 1e-4  # Normalized to unit sphere


def test_embedding_caching_and_provenance(esm_service, tmp_path):
    seq = "MAFLNGQLVTGAAQGIGRAYALALAKAGAQVVVNDLT"
    emb1, hash1 = esm_service.get_embedding(seq, target_id="TARGET-001")

    # Verify cached files on disk
    cache_tensor = tmp_path / f"TARGET-001_{hash1[:16]}.pt"
    cache_meta = tmp_path / f"TARGET-001_{hash1[:16]}_meta.json"
    assert cache_tensor.exists()
    assert cache_meta.exists()

    with open(cache_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["embedding_dim"] == 320
    assert "method" in meta
    assert "provenance_note" in meta
    assert "is_surrogate" in meta
    assert "extraction_date" in meta
    assert "device" in meta
    assert "processing_configuration" in meta

    # Second call must read from cache
    emb2, hash2 = esm_service.get_embedding(seq, target_id="TARGET-001")
    assert torch.equal(emb1, emb2)
    assert hash1 == hash2


def test_esm_service_status(esm_service):
    status = esm_service.get_service_status()
    assert "active_engine" in status
    assert status["embedding_dim"] == 320
    assert "huggingface_transformers_installed" in status
