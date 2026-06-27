"""Hand-reviewed label corrections for OCR'd QuizLink answers (HANDOUT v2 §5.5).

These are class labels for a medical recognition model, so they must be exactly
right. Each entry was reviewed by eye against the anatomy:

* a string value  -> the OCR text is a recoverable misspelling; map to the
  correct canonical structure name (e.g. "lliac crest" -> "iliac crest").
* ``None``         -> the OCR text is unrecoverable garbage or too ambiguous to
  assign to a structure with confidence; DROP the triple (a wrong class label is
  worse than a missing one).

Keys are the *normalized* label (lowercase, abbreviations expanded), i.e. exactly
what appears in ``triples.jsonl``. Applied in :func:`scalpel.data.clean.clean_dataset`
before canonicalization, so it survives a full rebuild.
"""

from __future__ import annotations

# recoverable OCR misspellings -> correct anatomical name
_FIX = {
    "alse vocal fold": "false vocal fold",
    "aryngeal artery": "laryngeal artery",
    "aryngeal nerve": "laryngeal nerve",
    "aryrgeel nerve": "laryngeal nerve",
    "aterventrieslar branch": "interventricular branch",
    "coronary ins": "coronary sinus",
    "external lliac artery": "external iliac artery",
    "falx cerebri cut": "falx cerebri",
    "ggreater palatine nerve": "greater palatine nerve",
    "hemi azygos vein": "hemiazygos vein",
    "hypoglossal nerve cn xiil": "hypoglossal nerve cn xii",
    "inferior vena": "inferior vena cava",
    "l bronchial arteries cut": "l bronchial arteries",
    "lateral mallelous": "lateral malleolus",
    "lleal arteries": "ileal arteries",
    "lleocolic artery": "ileocolic artery",
    "lliac crest": "iliac crest",
    "llio inguinal nerve": "ilioinguinal nerve",
    "lliohypogastric nerve": "iliohypogastric nerve",
    "lliolumbar artery": "iliolumbar artery",
    "lliopsoas muscle": "iliopsoas muscle",
    "maxillary nerve v5": "maxillary nerve v2",     # maxillary = CN V2, not "V5"
    "nancreativodaedenal artery": "pancreaticoduodenal artery",
    "nasopalatine nerve v5": "nasopalatine nerve v2",
    "omporal bone": "temporal bone",
    "optic nerve cn il": "optic nerve cn ii",
    "oulmonay vein": "pulmonary vein",
    "sacro iliac joints": "sacroiliac joints",
    "sancreaticodundenal artery": "pancreaticoduodenal artery",
    "slossopharyngeal nerve": "glossopharyngeal nerve",
    "sohenold bone": "sphenoid bone",
    "st maxillary sinvs": "maxillary sinus",
    "thyro arytenoid muscle": "thyroarytenoid muscle",
}

# unrecoverable OCR garbage / too ambiguous to assign a structure -> drop
_DROP = {
    "abdoninn muscle", "actimmal sland", "ae artery", "ae colateral",
    "alee oatrulata", "alig ie", "am artery tendon", "ambitieal rae",
    "amblifeal fold", "ane nace superior", "ant vor bel ly", "artery nsverse",
    "asion ct artery", "ate ee", "ateneetal muscles", "au cul artery",
    "aunguloferppersl nerve", "ayn 5 7 i", "ba sypogasins ns", "be feolal",
    "be lcncen", "bicep rac", "ca tilsg cut", "ce artery", "distal fadlonsines",
    "ee essa", "eo eetoalbe", "er artery", "estiha li", "et ell", "geena neveh",
    "gelate ganglion", "hashes rand", "he dee", "heaeaienealestl muscle",
    "heey on pemmens", "hnea seal", "i fil", "iavey lare", "ie cco artery ving ris",
    "ine nirlr ieera", "inferior opens", "ininsic me of", "interior sapita",
    "internal ureth ral", "io eton", "lak is", "lorcet an", "nae latine",
    "nema na", "neninges artery", "nesenaic artery", "noe it", "nopatin duct",
    "nteresetn am", "ntorenetal wa", "oa rior belly", "obligue muscle", "oey ae",
    "of prenyols ri", "of she voit vinw s", "ooo oper", "oudendat artery",
    "p occ ce", "pdieall pata", "pe artery inca", "pe renal fat eep",
    "pectoral alot muscle", "perc ae artery", "posh eee", "presse te",
    "r or stoner", "scnnon ey wv", "se eflectod", "se lbe", "se vembrance",
    "sicrnoi sinus", "slate wolatine", "sntehor ceslone on", "solu la",
    "sooo deep", "su mae pa or artery", "su sevior laryrceal nerve", "sulmoney oy",
    "superict ol sinus", "te mo ral", "te moral", "te roe", "tern o le on",
    "tu fecal", "valatin muscle", "vechue artery artery 3 i vem", "vee feced",
    "venous p seus",
}

# combined map consumed by clean.py: label -> corrected, or None to drop
OVERRIDES: dict[str, str | None] = {**_FIX, **{k: None for k in _DROP}}
