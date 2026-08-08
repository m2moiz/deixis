"""frames.py: the scoring, the ranking, and the marks that come out.

Two layers, deliberately separate. change_scores and select_marks are pure
functions of an array and are tested on hand-built arrays where the right
answer is arithmetic. mark_video and extract_tile_grid run real ffmpeg against
videos ffmpeg synthesizes at test time, because a stubbed test of a subprocess
wrapper asserts only that the stub was called.

The negative controls are the point of this file. A change detector that marks
everything and a change detector that marks nothing both LOOK like they work:
one returns a full budget, the other returns a tidy short list. The tests that
distinguish them are `test_static_video_yields_no_marks` (a video where nothing
happens must produce zero marks -- if this passes while the others do too, the
scoring is not merely present but discriminating) and
`test_marks_land_on_the_real_transitions`, which builds a video whose cuts are
at known seconds and demands the marks be there and not elsewhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from deixis import media
from deixis.frames import (
    DEFAULT_DELTA,
    Mark,
    change_scores,
    main,
    mark_video,
    select_marks,
    with_marks,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _tiles(*frames: list[list[int]]) -> np.ndarray:
    return np.array(frames, dtype=np.uint8)


# --------------------------------------------------------------------------
# change_scores
# --------------------------------------------------------------------------


def test_identical_frames_score_zero() -> None:
    tiles = _tiles([[10, 10], [10, 10]], [[10, 10], [10, 10]])
    assert change_scores(tiles).tolist() == [0]


def test_score_counts_tiles_over_delta_not_frames() -> None:
    # Three of four tiles move by 100, one stays put.
    tiles = _tiles([[0, 0], [0, 0]], [[100, 100], [100, 0]])
    assert change_scores(tiles, delta=8).tolist() == [3]


def test_a_move_exactly_at_delta_does_not_count() -> None:
    # The comparison is strictly greater, and a boundary that silently flips
    # would shift every score by the number of tiles sitting on it.
    tiles = _tiles([[0]], [[DEFAULT_DELTA]])
    assert change_scores(tiles, delta=DEFAULT_DELTA).tolist() == [0]
    tiles = _tiles([[0]], [[DEFAULT_DELTA + 1]])
    assert change_scores(tiles, delta=DEFAULT_DELTA).tolist() == [1]


def test_a_downward_move_counts_as_much_as_an_upward_one() -> None:
    up = change_scores(_tiles([[10]], [[250]]))
    down = change_scores(_tiles([[250]], [[10]]))
    assert up.tolist() == down.tolist() == [1]


def test_a_large_darkening_is_not_lost_to_an_unsigned_wrap() -> None:
    """The subtraction must be done in a signed type.

    248 -> 0 is the largest change a tile can nearly make. Under uint8 the
    subtraction wraps to exactly 8, which is DEFAULT_DELTA, so the tile reads as
    unchanged -- a screen going black would score zero. The gentler 250 -> 10
    case above does NOT catch this: it wraps to 16, still over delta, so the
    count comes out right for the wrong reason.
    """
    assert change_scores(_tiles([[248]], [[0]]), delta=DEFAULT_DELTA).tolist() == [1]


def test_scores_are_one_shorter_than_frames() -> None:
    tiles = np.zeros((5, 2, 2), dtype=np.uint8)
    assert len(change_scores(tiles)) == 4


def test_a_single_frame_has_no_intervals() -> None:
    assert change_scores(np.zeros((1, 2, 2), dtype=np.uint8)).tolist() == []


def test_no_frames_at_all_is_not_an_error() -> None:
    assert change_scores(np.zeros((0, 2, 2), dtype=np.uint8)).tolist() == []


def test_a_flat_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected an"):
        change_scores(np.zeros((4, 4), dtype=np.uint8))


def test_a_negative_delta_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        change_scores(np.zeros((2, 2, 2), dtype=np.uint8), delta=-1)


# --------------------------------------------------------------------------
# select_marks
# --------------------------------------------------------------------------


def _scores(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int32)


def test_the_budget_is_a_ceiling() -> None:
    scores = _scores(*range(1, 21))
    marks = select_marks(scores, fps=1.0, budget=5, min_gap_s=0.0)
    assert len(marks) == 5


def test_the_highest_scores_win() -> None:
    scores = _scores(1, 9, 2, 8, 3)
    marks = select_marks(scores, fps=1.0, budget=2, min_gap_s=0.0)
    assert [m.score for m in marks] == [9, 8]


def test_marks_come_back_in_time_order_not_ranked_order() -> None:
    scores = _scores(5, 100, 50)
    marks = select_marks(scores, fps=1.0, budget=3, min_gap_s=0.0)
    assert [m.t for m in marks] == sorted(m.t for m in marks)
    assert [m.t for m in marks] == [1.0, 2.0, 3.0]


def test_the_timestamp_is_the_frame_after_the_change() -> None:
    """Interval i sits between frames i and i+1; the change is visible at i+1.

    Off by one here points every mark at the last frame before the thing
    happened, which is the frame that does not show it.
    """
    marks = select_marks(_scores(0, 42), fps=1.0, budget=1, min_gap_s=0.0)
    assert marks == [Mark(t=2.0, score=42)]


def test_fps_scales_the_timestamps() -> None:
    marks = select_marks(_scores(0, 42), fps=2.0, budget=1, min_gap_s=0.0)
    assert marks == [Mark(t=1.0, score=42)]


def test_the_gap_suppresses_a_neighbour_however_high_it_scores() -> None:
    # Adjacent seconds, both enormous. The gap must keep only the larger.
    marks = select_marks(_scores(100, 99, 0, 0, 0, 0, 50), fps=1.0, budget=3, min_gap_s=5.0)
    assert [m.score for m in marks] == [100, 50]


def test_without_a_gap_the_same_input_clusters() -> None:
    """The negative control for the gap: it must actually change the answer.

    A min_gap_s that did nothing would leave every other test here passing.
    """
    marks = select_marks(_scores(100, 99, 0, 0, 0, 0, 50), fps=1.0, budget=3, min_gap_s=0.0)
    assert [m.score for m in marks] == [100, 99, 50]


def test_a_gap_of_exactly_min_gap_is_wide_enough() -> None:
    """The boundary is inclusive.

    Two marks exactly min_gap_s apart are far enough apart by definition;
    rejecting them costs one mark per boundary case for no reason a reader
    could name.
    """
    scores = _scores(100, 0, 0, 0, 0, 90)
    assert len(select_marks(scores, fps=1.0, budget=2, min_gap_s=5.0)) == 2
    assert len(select_marks(scores, fps=1.0, budget=2, min_gap_s=5.001)) == 1


def test_the_gap_is_measured_in_seconds_not_in_frames() -> None:
    """min_gap_s must be converted through fps, not compared to an index.

    At 2 fps these two intervals are 4 samples but only 2.0 seconds apart, so a
    3-second gap must reject the second one. Comparing indices directly would
    accept it -- and every other gap test here runs at 1 fps, where the two
    readings coincide and the bug is invisible.
    """
    scores = _scores(100, 0, 0, 0, 90)
    assert len(select_marks(scores, fps=2.0, budget=2, min_gap_s=3.0)) == 1
    assert len(select_marks(scores, fps=2.0, budget=2, min_gap_s=1.5)) == 2


def test_zero_scores_are_never_marked() -> None:
    """A budget is a ceiling, not a quota.

    Padding it with intervals where nothing moved would put marks on frames
    identical to the one before them.
    """
    marks = select_marks(_scores(7, 0, 0, 0, 0), fps=1.0, budget=4, min_gap_s=0.0)
    assert [m.score for m in marks] == [7]


def test_all_zero_scores_yield_nothing() -> None:
    assert select_marks(_scores(0, 0, 0), fps=1.0, budget=10, min_gap_s=0.0) == []


def test_no_scores_yield_nothing() -> None:
    assert select_marks(_scores(), fps=1.0, budget=10, min_gap_s=0.0) == []


def test_ties_break_toward_the_earlier_frame() -> None:
    marks = select_marks(_scores(5, 5, 5), fps=1.0, budget=1, min_gap_s=0.0)
    assert marks == [Mark(t=1.0, score=5)]


def test_ties_break_the_same_way_at_a_size_that_changes_the_sort() -> None:
    """Determinism: two runs over one recording must not disagree.

    numpy's default argsort is introsort, which is stable by accident on small
    inputs -- the three-element case above passes under either kind, so it pins
    nothing. Above the insertion-sort cutoff the two diverge: on this input the
    stable sort takes index 5 and quicksort takes index 152, both scoring 3. A
    real recording has hundreds of tied intervals, so this is the size that
    matters, not the small one.
    """
    scores = np.concatenate([np.zeros(5, dtype=np.int32), np.full(300, 3, dtype=np.int32)])
    marks = select_marks(scores, fps=1.0, budget=1, min_gap_s=0.0)
    assert marks == [Mark(t=6.0, score=3)]


def test_a_zero_budget_returns_nothing() -> None:
    assert select_marks(_scores(9, 9), fps=1.0, budget=0, min_gap_s=0.0) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fps": 0.0}, "fps must be positive"),
        ({"fps": -1.0}, "fps must be positive"),
        ({"budget": -1}, "budget must be non-negative"),
        ({"min_gap_s": -1.0}, "min_gap_s must be non-negative"),
    ],
)
def test_nonsense_parameters_are_rejected(kwargs: dict[str, float], match: str) -> None:
    args: dict[str, float] = {"fps": 1.0, "budget": 5, "min_gap_s": 0.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        select_marks(_scores(1, 2, 3), **args)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------
# with_marks
# --------------------------------------------------------------------------


def test_marks_are_added_beside_the_sentences() -> None:
    payload = {"audio": "a.mov", "sentences": [{"start": 0.0}]}
    out = with_marks(payload, [Mark(t=1.5, score=9)], {"budget": 1})
    assert out["marks"] == [{"t": 1.5, "score": 9}]
    assert out["sentences"] == payload["sentences"]


def test_the_original_payload_is_left_alone() -> None:
    payload: dict[str, object] = {"sentences": []}
    with_marks(payload, [Mark(t=1.0, score=1)], {})
    assert "marks" not in payload


def test_the_parameters_travel_with_the_marks() -> None:
    out = with_marks({}, [], {"budget": 150, "fps": 1.0})
    assert out["marks_meta"] == {"budget": 150, "fps": 1.0}


# --------------------------------------------------------------------------
# The ffmpeg boundary and the end-to-end pass
# --------------------------------------------------------------------------


def _run_ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _mean_grey(image: Path) -> float:
    """Average brightness of a still image, read back through ffmpeg.

    Not extract_tile_grid: that runs an `fps` filter, and a still has no
    duration for it to sample, so it returns zero frames. Found by running it.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(image),
         "-vf", "scale=8:8,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return float(np.frombuffer(out, dtype=np.uint8).mean())


def _dimensions(image: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(image)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    """6 seconds of one unchanging colour."""
    dest = tmp_path / "static.mp4"
    _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=0x202020:s=320x240:d=6:r=10",
        "-pix_fmt", "yuv420p", str(dest),
    )
    return dest


@pytest.fixture
def cut_video(tmp_path: Path) -> Path:
    """Three 4-second colours end to end, so the cuts are at t=4 and t=8.

    Concatenated as encoded segments rather than drawn with a filter: the point
    is a file whose transitions are at seconds a human can state in advance,
    and the assertion below is against those two numbers.
    """
    parts: list[Path] = []
    for i, colour in enumerate(("0x101010", "0xE0E0E0", "0x404040")):
        part = tmp_path / f"part{i}.mp4"
        _run_ffmpeg(
            "-f", "lavfi", "-i", f"color=c={colour}:s=320x240:d=4:r=10",
            "-pix_fmt", "yuv420p", str(part),
        )
        parts.append(part)
    listing = tmp_path / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    dest = tmp_path / "cuts.mp4"
    _run_ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(dest))
    return dest


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    dest = tmp_path / "t.json"
    dest.write_text(json.dumps({"audio": "x.mov", "sentences": [{"start": 0.0, "end": 1.0}]}))
    return dest


@needs_ffmpeg
def test_extract_tile_grid_returns_one_frame_per_sampled_second(static_video: Path) -> None:
    tiles = media.extract_tile_grid(static_video, fps=1.0, width=8, height=6)
    assert tiles.shape == (6, 6, 8)
    assert tiles.dtype == np.uint8


@needs_ffmpeg
def test_extract_tile_grid_reports_progress(static_video: Path) -> None:
    seen: list[float] = []
    media.extract_tile_grid(static_video, fps=1.0, width=8, height=6, on_progress=seen.append)
    assert seen == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@needs_ffmpeg
def test_frames_do_not_alias_one_another(cut_video: Path) -> None:
    """Each sampled frame must carry its own pixels.

    frombuffer returns a view, so a reader that reused one buffer across reads
    would hand back N copies of the last frame -- every score zero, every
    recording static, and no error anywhere to say so.
    """
    tiles = media.extract_tile_grid(cut_video, fps=1.0, width=8, height=6)
    assert not np.array_equal(tiles[0], tiles[5])


@needs_ffmpeg
def test_an_audio_only_file_is_refused(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(wav))
    with pytest.raises(media.NoVideoStream, match="no video stream"):
        media.extract_tile_grid(wav, fps=1.0, width=8, height=6)


@needs_ffmpeg
def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        media.extract_tile_grid(tmp_path / "nope.mov", fps=1.0, width=8, height=6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fps": 0.0}, "fps must be positive"),
        ({"width": 0}, "grid must be positive"),
        ({"height": -1}, "grid must be positive"),
    ],
)
@needs_ffmpeg
def test_extract_tile_grid_rejects_nonsense(
    static_video: Path, kwargs: dict[str, float], match: str
) -> None:
    args: dict[str, float] = {"fps": 1.0, "width": 8, "height": 6, **kwargs}
    with pytest.raises(ValueError, match=match):
        media.extract_tile_grid(static_video, **args)  # pyright: ignore[reportArgumentType]


@needs_ffmpeg
def test_extract_frame_writes_the_frame_at_that_second(cut_video: Path, tmp_path: Path) -> None:
    """The index is only worth having if what it points at can be opened.

    The fixture is black for 0-4s and near-white for 4-8s, so a seek that
    landed in the wrong segment shows up as a brightness the assertion below
    can catch -- where "a file was written" could not.
    """
    early = media.extract_frame(cut_video, 1.0, tmp_path / "early.png")
    late = media.extract_frame(cut_video, 6.0, tmp_path / "late.png")
    assert _mean_grey(early) < 64 < 192 < _mean_grey(late)


@needs_ffmpeg
def test_extract_frame_can_scale(cut_video: Path, tmp_path: Path) -> None:
    dest = media.extract_frame(cut_video, 1.0, tmp_path / "small.png", width=64)
    assert _dimensions(dest) == (64, 48)  # the fixture is 320x240


@needs_ffmpeg
def test_a_timestamp_past_the_end_is_an_error_not_an_empty_file(
    cut_video: Path, tmp_path: Path
) -> None:
    """A seek past the end fails deep in the encoder and writes nothing.

    Without the `dest.exists()` half of the check, a caller would get a path
    back and discover the absence later, somewhere with no context.
    """
    dest = tmp_path / "nope.jpg"
    with pytest.raises(media.MediaError, match="past the end"):
        media.extract_frame(cut_video, 9999.0, dest)
    assert not dest.exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"t": -1.0}, "t must be non-negative"), ({"width": 0}, "width must be positive")],
)
@needs_ffmpeg
def test_extract_frame_rejects_nonsense(
    cut_video: Path, tmp_path: Path, kwargs: dict[str, float], match: str
) -> None:
    args: dict[str, float] = {"t": 1.0, **kwargs}
    t = args.pop("t")
    with pytest.raises(ValueError, match=match):
        media.extract_frame(cut_video, t, tmp_path / "x.jpg", **args)  # pyright: ignore[reportArgumentType]


@needs_ffmpeg
def test_extract_frame_refuses_an_audio_only_file(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(wav))
    with pytest.raises(media.NoVideoStream):
        media.extract_frame(wav, 1.0, tmp_path / "x.jpg")


@needs_ffmpeg
def test_static_video_yields_no_marks(static_video: Path, transcript: Path, tmp_path: Path) -> None:
    """THE NEGATIVE CONTROL.

    A recording where nothing happens must produce nothing. If this fails, a
    full budget of marks on a real recording proves only that the budget was
    spent.
    """
    out = tmp_path / "out.json"
    result = mark_video(static_video, transcript, out, fps=1.0, budget=50, min_gap_s=0.0)
    assert result["marks"] == []


@needs_ffmpeg
def test_marks_land_on_the_real_transitions(
    cut_video: Path, transcript: Path, tmp_path: Path
) -> None:
    """THE POSITIVE CONTROL, with the answer known before the run.

    The fixture cuts colour at t=4 and t=8 and nowhere else. Both must be
    marked, and -- because a detector that marks every second would also
    satisfy "both are marked" -- nothing else may be.
    """
    out = tmp_path / "out.json"
    result = mark_video(cut_video, transcript, out, fps=1.0, budget=50, min_gap_s=0.0)
    marked = sorted(m["t"] for m in result["marks"])
    assert marked == [4.0, 8.0]


@needs_ffmpeg
def test_the_transcript_survives_the_pass(
    cut_video: Path, transcript: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.json"
    result = mark_video(cut_video, transcript, out, fps=1.0)
    assert result["sentences"] == [{"start": 0.0, "end": 1.0}]
    assert result["audio"] == "x.mov"
    assert json.loads(out.read_text()) == result


@needs_ffmpeg
def test_the_meta_records_what_was_run(cut_video: Path, transcript: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    result = mark_video(cut_video, transcript, out, fps=1.0, budget=7, min_gap_s=2.0)
    meta = result["marks_meta"]
    assert meta["budget"] == 7
    assert meta["min_gap_s"] == 2.0
    assert meta["frames_sampled"] == 12


@needs_ffmpeg
def test_a_missing_transcript_is_refused(cut_video: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mark_video(cut_video, tmp_path / "nope.json", tmp_path / "out.json")


@needs_ffmpeg
def test_the_cli_writes_marks_in_place(cut_video: Path, transcript: Path) -> None:
    assert main([str(cut_video), "-t", str(transcript), "--budget", "50", "--min-gap", "0"]) == 0
    written = json.loads(transcript.read_text())
    assert sorted(m["t"] for m in written["marks"]) == [4.0, 8.0]


@needs_ffmpeg
def test_the_cli_honours_an_explicit_output(
    cut_video: Path, transcript: Path, tmp_path: Path
) -> None:
    out = tmp_path / "elsewhere.json"
    assert main([str(cut_video), "-t", str(transcript), "-o", str(out)]) == 0
    assert "marks" in json.loads(out.read_text())
    assert "marks" not in json.loads(transcript.read_text())
