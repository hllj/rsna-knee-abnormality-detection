"""
Extract 12 soft labels from each (multilingual) radiology report with an LLM.

Host ruling (discussion 733965): commercial LLM APIs are allowed for this. Any OpenAI-compatible
endpoint works: OpenAI, OpenRouter, or a local vLLM / llama.cpp server running Qwen / Gemma.

    export OPENAI_API_KEY=...            # or OPENAI_BASE_URL=http://localhost:8000/v1 for vLLM
    python -m src.labeling.llm_label --model <model-id> --out data/labels/my_llm_labels.csv
    python scripts/eval_labels.py data/labels/my_llm_labels.csv

Design choices from the forum:
  * ask for a probability, not 0/1 — hedged reports ("suspected tear") become ~0.7-0.8
  * make "not mentioned" a first-class answer; it is written as 0.5 + verdict UNK so
    src/labeling/blend.py can apply per-finding silence priors
  * give the model the annotators' exact criteria (moderate/large effusion only, >50% ACL
    fibres, etc.): gold labels grade borderline findings as negative
  * don't tune the prompt on the 58 gold reports; it overfits the prompt to them
"""
import argparse
import asyncio
import json
import os

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.constants import ID_COL, LABELS

CRITERIA = """Label definitions used by the annotating MSK radiologists (borderline -> negative):
- ACL: high-grade partial (>50% fibres) or full-thickness ACL tear. Mild signal change, degeneration, mucoid thickening, or old reconstructed graft intact -> negative.
- MCL: high-grade partial or complete ACUTE MCL tear with fibre disruption and edema. Low-grade sprain, chronic/remote changes -> negative.
- Medial Meniscus: tear reaching the articular surface (on >=2 images) or morphologic abnormality (truncated, diminutive, displaced fragment, bucket-handle, root tear, maceration). Intrasubstance degeneration (grade 1-2 signal) -> negative.
- Lateral Meniscus: same criteria for the lateral meniscus.
- Medial OA: moderate/large (~>=1 cm) area of high-grade (>50% thickness) cartilage loss in the medial tibiofemoral compartment.
- Lateral OA: same, lateral tibiofemoral compartment.
- PF OA: same, patellofemoral compartment (patella / trochlea). Mild chondromalacia (grade 1-2) -> negative.
- Effusion: MODERATE or LARGE joint effusion. Small / trace / physiological fluid -> negative.
- Synovitis: synovial inflammation / thickening (incl. Hoffa synovitis, synovial proliferation).
- Baker's: moderate or large popliteal (Baker) cyst. Small cyst -> negative.
- Contusion: bone contusion / bone bruise / bone marrow edema from impact, without a discrete fracture line.
- Fracture: acute cortical break or fracture line (incl. osteochondral / impaction / avulsion fracture)."""

SYSTEM = f"""You are a musculoskeletal radiologist converting knee MRI reports (any language) into labels.
{CRITERIA}

For EACH of the 12 findings return:
  "status": one of "present", "absent", "not_mentioned"
  "p": probability (0-1) that the finding meets the definition above given the report.
Use "not_mentioned" only when the report says nothing that bears on the finding (a normal statement
about the structure counts as "absent"). Hedged language ("probable", "cannot exclude") -> p 0.5-0.8.
Answer with JSON only: {{"<finding>": {{"status": ..., "p": ...}}, ...}} using exactly these keys:
{json.dumps(LABELS)}"""


def parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1]
    d = json.loads(text[text.find("{"): text.rfind("}") + 1])
    row = {}
    for c in LABELS:
        v = d.get(c, {}) or {}
        status = str(v.get("status", "not_mentioned")).lower()
        p = float(v.get("p", 0.5))
        row[c] = 0.5 if status == "not_mentioned" else min(1.0, max(0.0, p))
        row[f"{c}__verdict"] = "UNK" if status == "not_mentioned" else status
    return row


async def label_one(client, sem, model, uid, report, temperature):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": report}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return {ID_COL: uid, **parse(r.choices[0].message.content)}
            except Exception as e:  # rate limits / malformed JSON: back off and retry
                if attempt == 3:
                    print(f"[fail] {uid}: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)


async def run(a):
    train = pd.read_csv(a.train, dtype={ID_COL: str})
    cache = a.out + ".jsonl"
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            r = json.loads(line)
            done[r[ID_COL]] = r
    todo = train[~train[ID_COL].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(a.concurrency)
    tasks = [label_one(client, sem, a.model, u, rep, a.temperature) for u, rep in zip(todo[ID_COL], todo["Report"])]
    with open(cache, "a") as f:
        for coro in tqdm_asyncio.as_completed(tasks):
            r = await coro
            if r:
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[r[ID_COL]] = r
    pd.DataFrame(list(done.values())).to_csv(a.out, index=False)
    print(f"wrote {len(done)} rows -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--model", required=True, help="e.g. an OpenAI model id, or the served name of a local vLLM model")
    ap.add_argument("--out", default="data/labels/my_llm_labels.csv")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
