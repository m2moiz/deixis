"""_clock renders a duration for a human reading a progress line.

Every expected string here was produced by running the real function; none was
computed by hand.
"""

from __future__ import annotations

import pytest

# Reaching for a private name is the point: _clock is internal to transcribe and
# these tests are its unit tests, so there is no public surface to target.
from deixis.transcribe import _clock  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "--:--"),
        (0.0, "0:00"),
        (59.0, "0:59"),
        (59.9, "0:59"),
        (60.0, "1:00"),
        (90.9, "1:30"),
        (3599.0, "59:59"),
        (3600.0, "1:00:00"),
        (3661.0, "1:01:01"),
        (4427.028, "1:13:47"),
    ],
    ids=[
        "none",
        "zero",
        "sub_minute",
        "truncates_not_rounds",
        "minute_boundary",
        "non_integer",
        "hour_minus_one",
        "hour_boundary",
        "hour_plus",
        "real_total",
    ],
)
def test_clock_formats(seconds: float | None, expected: str) -> None:
    assert _clock(seconds) == expected


def test_clock_distinguishes_zero_from_unknown() -> None:
    """A finished-at-zero run and a not-yet-known ETA must not look alike.

    `if not seconds` in place of `if seconds is None` would collapse these.
    """
    assert _clock(0.0) != _clock(None)
