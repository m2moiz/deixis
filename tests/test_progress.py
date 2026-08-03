"""Progress is the arithmetic behind every number a user sees during a run.

The real-run values come from an observed transcription of a 4427.028s
recording: 750s of audio done after 21.5s of wall clock.
"""

import pytest

from deixis.transcribe import Progress, render_bar

# One real observed sample from a live run.
REAL = Progress(audio_done_s=750.0, audio_total_s=4427.028, elapsed_s=21.5)


def test_fraction_is_audio_done_over_total():
    assert REAL.fraction == pytest.approx(0.16941388218009915)


def test_speed_is_realtime_multiple():
    assert REAL.speed == pytest.approx(34.883720930232556)


def test_eta_is_remaining_audio_at_current_speed():
    assert REAL.eta_s == pytest.approx(105.408136)


def test_zero_elapsed_yields_no_speed_and_no_eta():
    """The first callback can fire before the clock has advanced.

    Without the `if self.elapsed_s` guard this raises ZeroDivisionError before
    a single progress line is ever printed.
    """
    p = Progress(audio_done_s=10.0, audio_total_s=100.0, elapsed_s=0.0)
    assert p.fraction == 0.1
    assert p.speed == 0.0
    assert p.eta_s is None


def test_zero_total_yields_no_fraction_and_no_eta():
    """An empty or silent file reports a total of zero samples."""
    p = Progress(audio_done_s=0.0, audio_total_s=0.0, elapsed_s=5.0)
    assert p.fraction == 0.0
    assert p.speed == 0.0
    assert p.eta_s is None


def test_eta_is_none_before_any_audio_is_done():
    """speed == 0 must short-circuit; `speed < 0` instead of `<= 0` divides by zero."""
    p = Progress(audio_done_s=0.0, audio_total_s=100.0, elapsed_s=5.0)
    assert p.speed == 0.0
    assert p.eta_s is None


def test_done_beyond_total_is_not_clamped():
    """Pins today's behaviour so any future clamp is a deliberate change.

    Not reachable from parakeet-mlx, which passes min(start + chunk, len(audio)),
    but nothing in Progress forbids it.
    """
    p = Progress(audio_done_s=120.0, audio_total_s=100.0, elapsed_s=10.0)
    assert p.fraction == 1.2
    assert p.speed == 12.0
    assert p.eta_s == pytest.approx(-1.6666666666666667)


def test_render_bar_shows_absolute_durations_not_just_a_percentage():
    """The golden line for a real run.

    On the shipped samples-as-seconds bug this same Progress rendered
    '12:30' as '3333:20:00' and '1:13:47' as '19675:40:48', while the bar,
    the percentage and the ETA were byte-identical. The absolute clocks and
    the speed are the only fields that catch a units error.
    """
    assert render_bar(REAL, "running") == (
        "   running [####--------------------]  17%  "
        "12:30/1:13:47 audio  elapsed 0:21  eta 1:45  34.9x"
    )
