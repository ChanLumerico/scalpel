"""Label normalization + closed vocabulary 𝒱 (HANDOUT v2 §5.5).

QuizLink answer text (OCR'd from the baked label box) is noisy and full of
abbreviations/state descriptors ("m.", "a.", "v.", "n.", "(reflected)",
"(cut)"). :class:`Vocab` normalizes it to a canonical string and maps it to a
closed-vocabulary integer index; the vocabulary is persistable so train and
eval share the same 𝒱.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# parenthetical state descriptors that are not part of the structure id
_STATE = re.compile(r"\((?:reflected|cut|retracted|removed|partially[^)]*)\)", re.I)
# standalone QuizLink abbreviations -> full word (matched per token after de-punct)
_ABBR = {
    "m": "muscle", "a": "artery", "v": "vein", "n": "nerve", "lig": "ligament",
    "t": "tendon", "br": "branch", "gl": "gland",
    # doubled forms are the Latin plural ("mm." = musculi = muscles)
    "mm": "muscles", "aa": "arteries", "vv": "veins", "nn": "nerves",
}


class Vocab:
    """Normalize raw label text and map it into a closed vocabulary."""

    def __init__(self, names: list[str] | None = None):
        self._name_to_idx: dict[str, int] = {}
        for n in names or []:
            self.index(n)

    # -- normalization ------------------------------------------------------
    @staticmethod
    def normalize(raw: str) -> str:
        s = raw.strip().lower().replace("\n", " ")
        s = _STATE.sub("", s)                       # drop "(reflected)" etc.
        s = re.sub(r"[^a-z0-9 ]+", " ", s)          # strip punctuation ("m." -> "m")
        toks = [_ABBR.get(t, t) for t in s.split()]  # expand standalone abbreviations
        return " ".join(toks).strip()

    # -- closed vocabulary --------------------------------------------------
    def index(self, name: str) -> int:
        """Map a (possibly raw) name to its index, assigning a new one if unseen."""
        key = self.normalize(name)
        if key not in self._name_to_idx:
            self._name_to_idx[key] = len(self._name_to_idx)
        return self._name_to_idx[key]

    def get(self, name: str) -> int | None:
        return self._name_to_idx.get(self.normalize(name))

    @property
    def names(self) -> list[str]:
        return sorted(self._name_to_idx, key=self._name_to_idx.get)

    def __len__(self) -> int:
        return len(self._name_to_idx)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._name_to_idx, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        v = cls()
        v._name_to_idx = json.loads(Path(path).read_text())
        return v
