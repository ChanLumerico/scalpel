"""Experiment 035 (build) — generate the human-ceiling annotation packet.

Produces a BLIND, evaluator-agnostic quiz packet that shows a human EXACTLY what the
model sees (full image resized to 518px + a pin marker — NOT a crop), and records the
model's own per-pin prediction for the later error-overlap analysis.

Outputs under data/human_ceiling/ (gitignored — cadaver imagery, §3):
  packet/<id>.png        pin-marked 518px image (model-input replica)
  quiz.html              self-contained blind quiz (215-way MC + free-recall + 모름)
  manifest_quiz.json     [{id}]                       — what the quiz reads (NO answers)
  manifest_answers.jsonl {id,true,tissue,model_*}     — for scoring only (answers hidden)
  vocab.json             215 class names (MC autocomplete)

Model prediction per pin = specimen-LOO exemplar 1-NN (same page excluded), matching
the canonical eval. Sample n≈100, stratified to over-represent confusable tissues
(artery/vein/nerve) and balanced model-correct vs model-wrong so the 2×2 cells fill.

    .venv/bin/python scripts/human_ceiling_build.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from eval_appearance import load_core, embed  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

OUT = Path("data/human_ceiling")
N_TARGET = 100
TARGETS = {"artery": 22, "vein": 13, "nerve": 22, "muscle": 22, "bone": 9, "other": 12}
SEED = 0


def tissue(label):
    s = label.lower()
    for kw, t in [("artery", "artery"), ("arteria", "artery"), ("vein", "vein"), ("vena", "vein"),
                  ("nerve", "nerve"), ("nervus", "nerve"), ("muscle", "muscle"), ("musculus", "muscle"),
                  ("bone", "bone")]:
        if kw in s:
            return t
    return "other"


def loo_predict(Z, spec, Y):
    """Per-pin specimen-LOO exemplar 1-NN → (top1, top5, covered) per pin."""
    S = Z @ Z.T
    N = len(Y)
    out = []
    for i in range(N):
        cls_best = {}
        si = spec[i]
        row = S[i]
        for j in range(N):
            if spec[j] == si:
                continue
            l = Y[j]
            v = row[j]
            if v > cls_best.get(l, -2.0):
                cls_best[l] = v
        if not cls_best:
            out.append((None, [], False)); continue
        ranked = sorted(cls_best, key=lambda l: -cls_best[l])
        out.append((ranked[0], ranked[:5], Y[i] in cls_best))
    return out


def stratified_sample(cand, tis, top1ok):
    rng = np.random.default_rng(SEED)
    by_t = collections.defaultdict(lambda: {"ok": [], "no": []})
    for i in cand:
        by_t[tis[i]]["ok" if top1ok[i] else "no"].append(i)
    chosen = []
    for t, k in TARGETS.items():
        ok = by_t[t]["ok"][:]; no = by_t[t]["no"][:]
        rng.shuffle(ok); rng.shuffle(no)
        half = k // 2
        take = no[:half] + ok[:k - half]
        if len(take) < k:                                   # backfill from whichever remains
            pool = no[half:] + ok[k - half:]
            take += pool[:k - len(take)]
        chosen += take[:k]
    rng.shuffle(chosen)
    return chosen


def draw_pin(arr, x, y):
    cv2.circle(arr, (x, y), 17, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(arr, (x, y), 17, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(arr, (x, y), 2, (255, 0, 255), -1, cv2.LINE_AA)
    return arr


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    spec = [f'{r["src"]}#{r["page"]}' for r in core]
    vocab = sorted(set(Y))
    print(f"core {len(core)}/{len(vocab)} | device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding..."); Z = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    preds = loo_predict(Z, spec, Y)
    tis = [tissue(l) for l in Y]
    covered = [p[2] for p in preds]
    top1ok = [p[0] == Y[i] for i, p in enumerate(preds)]
    top5ok = [Y[i] in p[1] for i, p in enumerate(preds)]
    cand = [i for i in range(len(core)) if covered[i]]
    print(f"covered {len(cand)} | model top1 {100*np.mean([top1ok[i] for i in cand]):.1f} "
          f"top5 {100*np.mean([top5ok[i] for i in cand]):.1f} (on covered)")

    chosen = stratified_sample(cand, tis, top1ok)
    tc = collections.Counter(tis[i] for i in chosen)
    oc = collections.Counter(("ok" if top1ok[i] else "no") for i in chosen)
    print(f"sampled n={len(chosen)} | tissue {dict(tc)} | model {dict(oc)}")

    (OUT / "packet").mkdir(parents=True, exist_ok=True)
    quiz, answers = [], []
    img_cache = {}
    for k, i in enumerate(chosen):
        r = core[i]; qid = f"q{k:03d}"
        if r["image"] not in img_cache:
            im = Image.open(base / r["image"]).convert("RGB")
            img_cache[r["image"]] = (np.asarray(im.resize((S, S)), np.uint8).copy(), im.size)
        arr0, (w, h) = img_cache[r["image"]]
        arr = arr0.copy()
        x, y = int(r["q"][0] * S / w), int(r["q"][1] * S / h)
        draw_pin(arr, x, y)
        cv2.imwrite(str(OUT / "packet" / f"{qid}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        quiz.append({"id": qid})
        answers.append({"id": qid, "true": Y[i], "tissue": tis[i],
                        "model_top1": preds[i][0], "model_top5": preds[i][1],
                        "model_top1_correct": bool(top1ok[i]), "model_top5_correct": bool(top5ok[i])})

    (OUT / "manifest_quiz.json").write_text(json.dumps(quiz, ensure_ascii=False), encoding="utf-8")
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "manifest_answers.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    html = _HTML.replace("{{VOCAB}}", json.dumps(vocab, ensure_ascii=False)) \
               .replace("{{ITEMS}}", json.dumps(quiz, ensure_ascii=False))
    (OUT / "quiz.html").write_text(html, encoding="utf-8")
    print(f"wrote packet ({len(chosen)} items) -> {OUT}/  | open {OUT}/quiz.html in a browser")
    return 0


_HTML = r"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>SCALPEL — human ceiling (blind)</title>
<style>
 body{font-family:system-ui,Apple SD Gothic Neo,sans-serif;max-width:680px;margin:18px auto;padding:0 14px;color:#1a1a1a}
 img{width:100%;max-width:560px;border:1px solid #ccc;border-radius:6px;display:block;margin:8px 0}
 .bar{height:6px;background:#eee;border-radius:3px;overflow:hidden;margin:6px 0 14px}
 .bar>i{display:block;height:100%;background:#c0457a}
 label{font-weight:600;display:block;margin:12px 0 4px}
 input[type=text]{width:100%;padding:9px;font-size:15px;border:1px solid #bbb;border-radius:5px;box-sizing:border-box}
 .row{display:flex;gap:8px;align-items:center;margin-top:18px}
 button{padding:10px 18px;font-size:15px;border:0;border-radius:6px;background:#c0457a;color:#fff;cursor:pointer}
 button.sec{background:#eee;color:#333}
 .hint{color:#777;font-size:13px;font-weight:400}
 .mc input{margin-bottom:6px}
</style></head><body>
<h3>SCALPEL 해부 구조 식별 (블라인드)</h3>
<div class=hint>핀(자홍색 원)이 가리키는 구조를 식별하세요. 정답·점수는 보이지 않습니다.</div>
<div class=bar><i id=pg></i></div>
<div id=meta class=hint></div>
<img id=im>
<label>1) 자유 회상 <span class=hint>— 목록 없이, 기억나는 대로 (모르면 비워두세요)</span></label>
<input type=text id=free autocomplete=off>
<label>2) 객관식 <span class=hint>— 목록에서 1~5순위 (1순위 = 최선의 답)</span></label>
<div class=mc>
 <input type=text class=mc1 list=vocab placeholder="1순위 (필수)" autocomplete=off>
 <input type=text class=mc1 list=vocab placeholder="2순위" autocomplete=off>
 <input type=text class=mc1 list=vocab placeholder="3순위" autocomplete=off>
 <input type=text class=mc1 list=vocab placeholder="4순위" autocomplete=off>
 <input type=text class=mc1 list=vocab placeholder="5순위" autocomplete=off>
</div>
<datalist id=vocab></datalist>
<div class=row>
 <button class=sec id=dk>모름 / 건너뛰기</button>
 <button id=nx>다음 →</button>
 <span class=hint id=cnt></span>
</div>
<script>
const VOCAB={{VOCAB}}, ITEMS={{ITEMS}};
const dl=document.getElementById('vocab');
VOCAB.forEach(v=>{const o=document.createElement('option');o.value=v;dl.appendChild(o);});
let idx=0, t0=Date.now(); const ans={};
const $=s=>document.querySelector(s), mcs=()=>[...document.querySelectorAll('.mc1')];
function show(){
 const it=ITEMS[idx];
 $('#im').src='packet/'+it.id+'.png';
 $('#free').value=''; mcs().forEach(e=>e.value='');
 $('#pg').style.width=(100*idx/ITEMS.length)+'%';
 $('#cnt').textContent=(idx+1)+' / '+ITEMS.length;
 $('#meta').textContent='문항 '+it.id;
 t0=Date.now(); $('#free').focus();
}
function save(dk){
 const it=ITEMS[idx];
 ans[it.id]={id:it.id, free:$('#free').value.trim(),
   mc:mcs().map(e=>e.value.trim()).filter(Boolean), dontknow:!!dk, ms:Date.now()-t0};
}
function next(dk){
 save(dk); idx++;
 if(idx>=ITEMS.length){finish();return;}
 show();
}
function finish(){
 const blob=new Blob([JSON.stringify(Object.values(ans),null,1)],{type:'application/json'});
 const u=URL.createObjectURL(blob), a=document.createElement('a');
 a.href=u; a.download='answers.json'; a.click();
 document.body.innerHTML='<h3>완료 — answers.json 다운로드됨</h3>'+
   '<div class=hint>이 파일을 data/human_ceiling/ 에 두고 채점 스크립트를 실행하세요.</div>';
}
$('#nx').onclick=()=>next(false);
$('#dk').onclick=()=>next(true);
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.tagName!=='TEXTAREA'){e.preventDefault();next(false);}});
show();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
