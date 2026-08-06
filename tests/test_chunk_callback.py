"""The sample-to-second conversion that already shipped one bug.

parakeet-mlx calls chunk_callback(end, len(audio_data)) with SAMPLE indices
(parakeet_mlx/parakeet.py:182-186). The first release read them as seconds and
wrote audio_total_s: 70832448 for a 74-minute file. Crucially, `fraction` and
`eta_s` were CORRECT throughout, because units cancel in a ratio -- so only
assertions on absolute values can catch this.
"""

from __future__ import annotations

import pytest

# _make_chunk_callback is private to deixis.transcribe by design -- it is a
# construction detail with no public caller -- but its sample-to-second
# conversion is exactly what this module exists to pin, so the test must reach
# it directly. Renaming it public would change the source, not the test.
from deixis.transcribe import Progress, _make_chunk_callback  # pyright: ignore[reportPrivateUsage]

RATE = 16_000
REAL_DONE_S = 750.0
REAL_TOTAL_S = 4427.028


def test_chunk_callback_reports_seconds_not_samples() -> None:
    seen: list[Progress] = []
    cb = _make_chunk_callback(rate=RATE, clock=lambda: 21.5, emit=seen.append)

    cb(REAL_DONE_S * RATE, REAL_TOTAL_S * RATE)

    p = seen[-1]
    assert p.audio_done_s == pytest.approx(REAL_DONE_S)
    assert p.audio_total_s == pytest.approx(REAL_TOTAL_S)
    assert p.speed == pytest.approx(34.883720930232556)


def test_chunk_callback_totals_stay_on_a_plausible_human_scale() -> None:
    """Tripwire: 70832448 'seconds' is 2.2 years of audio."""
    seen: list[Progress] = []
    cb = _make_chunk_callback(rate=RATE, clock=lambda: 21.5, emit=seen.append)

    cb(REAL_DONE_S * RATE, REAL_TOTAL_S * RATE)

    assert seen[-1].audio_total_s < 86_400


@pytest.mark.parametrize("rate", [8_000, 16_000, 44_100])
def test_chunk_callback_honours_the_models_sample_rate(rate: int) -> None:
    """A hardcoded 16000 passes at 16kHz and is wrong everywhere else."""
    seen: list[Progress] = []
    cb = _make_chunk_callback(rate=rate, clock=lambda: 1.0, emit=seen.append)

    cb(10.0 * rate, 60.0 * rate)

    assert seen[-1].audio_done_s == pytest.approx(10.0)
    assert seen[-1].audio_total_s == pytest.approx(60.0)


def test_chunk_callback_stamps_elapsed_from_the_clock() -> None:
    seen: list[Progress] = []
    ticks = iter([2.0, 5.0])
    cb = _make_chunk_callback(rate=RATE, clock=lambda: next(ticks), emit=seen.append)

    cb(1.0 * RATE, 10.0 * RATE)
    cb(2.0 * RATE, 10.0 * RATE)

    assert [p.elapsed_s for p in seen] == [2.0, 5.0]


def test_chunk_callback_emits_once_per_chunk() -> None:
    seen: list[Progress] = []
    cb = _make_chunk_callback(rate=RATE, clock=lambda: 1.0, emit=seen.append)

    cb(1.0 * RATE, 10.0 * RATE)
    cb(2.0 * RATE, 10.0 * RATE)
    cb(3.0 * RATE, 10.0 * RATE)

    assert len(seen) == 3
