"""Triples -> few-shot gallery / eval set with specimen-level splits (v2 §5.4).

★ Triples from the same photo/page are strongly correlated, so train/test MUST
be split at the **specimen** level (here: per PDF page), never per triple, or
the evaluation leaks (handout §5.2 / §8.4). The gallery feeds
``PrototypicalHead.fit`` (via ``loops.fewshot_adapt``); the test set feeds
``loops.evaluate``. Both yield ``(PIL image, q, int_label)`` items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parse import Triple
from .vocab import Vocab


@dataclass
class Splits:
    train: list[Triple] = field(default_factory=list)
    test: list[Triple] = field(default_factory=list)


def _specimen(t: Triple) -> str:
    return f"{t.src}#{t.page}"  # one PDF page == one specimen


def build_splits(
    triples: list[Triple], by: str = "specimen", test_frac: float = 0.3, seed: int = 0
) -> Splits:
    """Split triples into train/test grouped by specimen (no page in both)."""
    import numpy as np

    if by != "specimen":
        raise ValueError("only specimen-level splitting is allowed (§5.2)")
    groups: dict[str, list[Triple]] = {}
    for t in triples:
        groups.setdefault(_specimen(t), []).append(t)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    test_keys = set(keys[:n_test])
    sp = Splits()
    for k in keys:
        (sp.test if k in test_keys else sp.train).extend(groups[k])
    return sp


def _items(triples: list[Triple], vocab: Vocab):
    """``(PIL image, (x, y), int_label)`` for the engine's gallery/test consumers."""
    from PIL import Image

    for t in triples:
        yield Image.fromarray(t.image), t.q, vocab.index(t.label)


def to_gallery(triples: list[Triple], vocab: Vocab) -> list:
    """Few-shot support set for ``loops.fewshot_adapt`` (no gradient)."""
    return list(_items(triples, vocab))


def to_testset(triples: list[Triple], vocab: Vocab) -> list:
    """Held-out eval set for ``loops.evaluate`` (specimen-disjoint from gallery)."""
    return list(_items(triples, vocab))
