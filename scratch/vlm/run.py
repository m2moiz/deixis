"""Run gemma-4-e2b-it-4bit over the frame set, one model load, timed per image."""

import json
import sys
import time

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/gemma-4-e2b-it-4bit"
PROMPT = (
    "Transcribe all text visible in this screenshot, then describe what "
    "application is shown and what the user is doing. Be exhaustive about the text."
)

images = sys.argv[1:]

t0 = time.perf_counter()
model, processor = load(MODEL)
config = model.config
load_s = time.perf_counter() - t0
print(f"model load: {load_s:.1f}s", file=sys.stderr)

results = []
for path in images:
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
    t = time.perf_counter()
    out = generate(
        model, processor, formatted, [path], max_tokens=400, verbose=False
    )
    elapsed = time.perf_counter() - t
    text = out.text if hasattr(out, "text") else str(out)
    print(f"{path}: {elapsed:.1f}s", file=sys.stderr)
    results.append({"image": path, "seconds": round(elapsed, 1), "output": text})

json.dump(
    {"model": MODEL, "load_seconds": round(load_s, 1), "results": results},
    open("/Users/moiz/Documents/code/jaano/scratch/vlm/raw_output.json", "w"),
    indent=2,
    ensure_ascii=False,
)
