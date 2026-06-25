"""SCALPEL - Synthetic-to-real Cadaveric Anatomy Localization via
Point-conditioned Expert Learning.

A point-conditioned, scene-graph reasoning model for gross-anatomy spot exams
(handout abstract). Estimates ``p(y | I, q)`` as a Product of Experts over an
appearance expert (frozen DINOv2 + point pooling + prototypical few-shot) and a
relational expert (segmentation -> scene graph -> R-GCN).

Heavy entry points (anything that drags in torch / torchvision / DINO) are
exposed through PEP 562 lazy imports so that importing :mod:`scalpel` or one of
its lightweight submodules does not pay for the heavy dependencies until a
heavy symbol is actually touched (handout §4.10, §8.7).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    # config (lightweight, always safe)
    "PipelineCfg",
    "default_device",
    # heavy entry points (lazy)
    "ScalpelPipeline",
    "DinoBackbone",
    "PointConditioner",
    "PatchSegmenter",
    "PrototypicalHead",
    "TemperatureScaler",
    "ProductOfExperts",
    "RelationalGNN",
    "LLMReasoner",
]

# symbol -> submodule that defines it (resolved on first access)
_LAZY: dict[str, str] = {
    "ScalpelPipeline": "scalpel.pipeline",
    "DinoBackbone": "scalpel.perception",
    "PointConditioner": "scalpel.perception",
    "PatchSegmenter": "scalpel.perception",
    "PrototypicalHead": "scalpel.heads",
    "TemperatureScaler": "scalpel.heads",
    "ProductOfExperts": "scalpel.heads",
    "RelationalGNN": "scalpel.relational_gnn",
    "LLMReasoner": "scalpel.llm_reasoner",
}

# config is cheap (no torch at import time) so expose it eagerly.
from .config import PipelineCfg, default_device  # noqa: E402


def __getattr__(name: str):  # PEP 562
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'scalpel' has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # help static analysers without triggering heavy imports
    from .heads import PrototypicalHead, ProductOfExperts, TemperatureScaler
    from .llm_reasoner import LLMReasoner
    from .perception import DinoBackbone, PatchSegmenter, PointConditioner
    from .pipeline import ScalpelPipeline
    from .relational_gnn import RelationalGNN

__version__ = "0.1.0"
