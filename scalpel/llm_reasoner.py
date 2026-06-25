"""Optional LLM-over-graph reasoning layer (handout §2.8, §4.7).

A *frozen* language/vision model is used only as a reasoning layer - never to
localize pixels (that is what the perception stack is for). Two externalisation
strategies, both keeping the LM away from pixel grounding:

* **Set-of-Mark** - draw the pin on the image (``mark_pin``) and let a VLM reason
  about the marked context.
* **Structured graph prompt** - serialise the pin's neighbourhood
  (``serialise_neighbourhood``) so an LLM reasons over *symbols*.

The backend is swappable (local MLX-VLM / API / stub), so this module never
hard-depends on any one provider. The :class:`StubBackend` keeps the smoke path
asset-free (handout §8.8).

Interface note (handout §8.9): :meth:`LLMReasoner.reason` takes ``shortlist`` and
must forward it to :func:`build_prompt` as ``candidate_shortlist`` - the names
differ on purpose; the smoke test guards against wiring them up wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .scene_graph import RELATION_NAMES, SceneGraph

_ANSWER_RE = re.compile(
    r"ANSWER:\s*(?P<ans>.+?)\s+CONF:\s*(?P<conf>[0-9]*\.?[0-9]+)", re.IGNORECASE
)


@runtime_checkable
class LMBackend(Protocol):
    """Anything that maps a (prompt, optional image) to text."""

    def generate(self, prompt: str, image=None) -> str: ...


class StubBackend:
    """Placeholder backend used when no real LM is configured (handout §4.7).

    Returns a clearly-marked placeholder that still ends with the enforced
    ``ANSWER: ... CONF: ...`` line so the parsing path stays exercised offline.
    """

    def generate(self, prompt: str, image=None) -> str:
        return (
            "[StubBackend] No language-model backend is configured; this is a "
            "placeholder response and performs no real reasoning.\n"
            "ANSWER: unknown  CONF: 0.00"
        )


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #
def mark_pin(img, q, r: int = 14, style: str = "arrow",
             angle_deg: float = 35.0, length: int = 95, color=(220, 20, 20)):
    """Set-of-Mark: mark pin ``q = (x, y)`` on a copy of ``img``.

    ``style="arrow"`` draws a thick red arrow pointing at q — the gross-anatomy
    spot-exam convention (the real exam marks the target with an arrow).
    ``style="circle"`` draws the original ringed crosshair.
    """
    import math

    from PIL import ImageDraw

    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = float(q[0]), float(q[1])
    if style == "arrow":
        a = math.radians(angle_deg)
        tx, ty = x + length * math.cos(a), y + length * math.sin(a)   # tail (outside)
        draw.line([tx, ty, x, y], fill=color, width=max(5, r // 2 * 2))
        head = r + 10
        p1 = (x + head * math.cos(a + 0.42), y + head * math.sin(a + 0.42))
        p2 = (x + head * math.cos(a - 0.42), y + head * math.sin(a - 0.42))
        draw.polygon([(x, y), p1, p2], fill=color)
    else:
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        draw.line([x - r - 4, y, x + r + 4, y], fill=color, width=2)
        draw.line([x, y - r - 4, x, y + r + 4], fill=color, width=2)
    return out


def serialise_neighbourhood(
    graph: SceneGraph, pin_idx: int, label_names: list[str]
) -> str:
    """Serialise the pin region's typed neighbourhood as text (handout §8.3).

    Reads every edge as "the pin is [rel] <neighbour>", so a central structure
    comes out as "deep to / medial to ..." rather than "lateral".
    """

    def lname(node: int) -> str:
        lab = graph.regions[node].label
        return label_names[lab] if lab < len(label_names) else f"structure_{lab}"

    pin_name = lname(pin_idx)
    by_rel: dict[int, list[str]] = {}
    for j, rel in graph.neighbors(pin_idx):
        by_rel.setdefault(rel, []).append(lname(j))

    lines = [f"The pin is on: {pin_name}."]
    if not by_rel:
        lines.append("It has no recorded neighbours.")
    for rel in sorted(by_rel):
        names = ", ".join(sorted(set(by_rel[rel])))
        lines.append(f"It is {RELATION_NAMES[rel]} {names}.")
    return "\n".join(lines)


def build_prompt(
    graph: SceneGraph,
    pin_idx: int,
    label_names: list[str],
    candidate_shortlist: list[str] | None = None,
) -> str:
    """Build the textual reasoning prompt for the LM (handout §4.7)."""
    neighbourhood = serialise_neighbourhood(graph, pin_idx, label_names)
    parts = [
        "You are an anatomy examiner identifying the structure under a pin in a "
        "gross-anatomy spot exam. Reason from the local appearance and the "
        "topological relations to the surrounding structures.",
        "",
        "Scene graph around the pin:",
        neighbourhood,
    ]
    if candidate_shortlist:
        parts += [
            "",
            "Restrict your answer to one of these candidates:",
            ", ".join(candidate_shortlist),
        ]
    parts += [
        "",
        "Respond with a one-line justification, then end with EXACTLY this line:",
        "ANSWER: <structure name>  CONF: <confidence 0-1>",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Reasoner                                                                    #
# --------------------------------------------------------------------------- #
def _parse_answer(text: str) -> tuple[str | None, float]:
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None, 0.0
    m = matches[-1]  # last enforced line wins
    conf = max(0.0, min(1.0, float(m.group("conf"))))
    return m.group("ans").strip(), conf


@dataclass
class LLMReasoner:
    """High-level verifier / standalone reasoner over the scene graph (§2.8)."""

    backend: LMBackend = field(default_factory=StubBackend)
    label_names: list[str] = field(default_factory=list)
    use_image: bool = True

    def reason(
        self,
        image,
        q,
        graph: SceneGraph,
        pin_idx: int,
        shortlist: list[str] | None = None,
    ) -> dict:
        """Reason about the pinned structure. Returns ``{raw, answer, confidence}``.

        ``shortlist`` is forwarded to :func:`build_prompt` as
        ``candidate_shortlist`` (handout §8.9 - the names intentionally differ).
        """
        prompt = build_prompt(
            graph, pin_idx, self.label_names, candidate_shortlist=shortlist
        )
        marked = mark_pin(image, q) if (image is not None and self.use_image) else None
        raw = self.backend.generate(prompt, image=marked)
        answer, confidence = _parse_answer(raw)
        return {"raw": raw, "answer": answer, "confidence": confidence}
