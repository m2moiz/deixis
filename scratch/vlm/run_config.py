"""Run one (model, prompt, max_tokens) configuration over the frame set.

run.py pinned the exact configuration measured on 2026-08-08 and is left alone so
that measurement stays reproducible. This script takes the same shape but lets
the three variables the 2026-08-08 write-up named as untested — model, prompt
wording, token budget — move independently, so a later run differs from the
baseline in a known way rather than an unknown one.

Decoding is left at mlx-vlm's default, which is greedy (`DEFAULT_TEMPERATURE`
is 0.0), matching the baseline: every reading is the model's argmax, so any
difference against the baseline is the configuration and not sampling noise.

Usage:
    python run_config.py --model M --prompt-file P --max-tokens N --out O IMG...
"""

import argparse
import json
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().strip()

    t0 = time.perf_counter()
    model, processor = load(args.model)
    config = model.config
    load_s = time.perf_counter() - t0
    print(f"model load: {load_s:.1f}s", flush=True)

    results = []
    for path in args.images:
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        t = time.perf_counter()
        out = generate(
            model, processor, formatted, [path], max_tokens=args.max_tokens, verbose=False
        )
        elapsed = time.perf_counter() - t
        text = out.text if hasattr(out, "text") else str(out)
        print(f"{path}: {elapsed:.1f}s, {len(text)} chars", flush=True)
        results.append({"image": path, "seconds": round(elapsed, 1), "output": text})

    Path(args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "load_seconds": round(load_s, 1),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
