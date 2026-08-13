"""Three timed variants on the same audio, to locate the regression."""
import time
from pathlib import Path

import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio

from dsj import checkpoint, chunking

AUDIO = Path("scratch/clip360.wav")
m = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
rate = m.preprocessor_config.sample_rate
audio = load_audio(AUDIO, rate, mx.bfloat16)
print(f"audio samples={len(audio)} ({len(audio)/rate:.0f}s)", flush=True)

def timed(label, fn):
    t = time.monotonic()
    r = fn()
    el = time.monotonic() - t
    print(f"{label:38s} {el:7.1f}s", flush=True)
    return r, el

# A: library's own transcribe (the 35.4x baseline path)
timed("A library transcribe()", lambda: m.transcribe(AUDIO, chunk_duration=120.0, overlap_duration=15.0))

# B: our loop, NO on_chunk callback at all
timed("B our loop, no callback", lambda: chunking.transcribe_chunked(
    m, audio, chunk_s=120.0, overlap_s=15.0))

# C: our loop, callback that only counts (no serialisation, no I/O)
n = [0]
timed("C our loop, counting callback", lambda: chunking.transcribe_chunked(
    m, audio, chunk_s=120.0, overlap_s=15.0,
    on_chunk=lambda d, ns, t, merged: n.__setitem__(0, n[0]+1)))

# D: our loop, real checkpoint write
ck = Path("scratch/prof.ckpt")
fp = checkpoint.fingerprint(AUDIO, len(audio), "mlx-community/parakeet-tdt-0.6b-v3", 120.0, 15.0)
timed("D our loop, real checkpoint write", lambda: chunking.transcribe_chunked(
    m, audio, chunk_s=120.0, overlap_s=15.0,
    on_chunk=lambda d, ns, t, merged: checkpoint.write_checkpoint(ck, fp, ns, merged)))
