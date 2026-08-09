"""The `deixis` command: does each verb reach the thing it names?

Small surface, but the one that decides whether any of this is usable. The
measurement that justifies frame retrieval (docs/do-marks-help.md) was only
possible because a Python snippet was hand-written into an agent's prompt; these
tests exist so the supported path cannot quietly stop working.

The dispatch tests assert ROUTING, not behaviour -- transcribe and mark are
covered by their own suites, and re-running real ASR here would buy nothing for
two minutes of wall clock.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from deixis.cli import main

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Two colours, 4 seconds each, so a seek can be checked for landing right."""
    parts: list[Path] = []
    for i, colour in enumerate(("0x101010", "0xE0E0E0")):
        part = tmp_path / f"p{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c={colour}:s=320x240:d=4:r=10", "-pix_fmt", "yuv420p", str(part)],
            check=True,
        )
        parts.append(part)
    listing = tmp_path / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    dest = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        check=True,
    )
    return dest


def _dimensions(image: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(image)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _mean_grey(image: Path) -> float:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(image),
         "-vf", "scale=4:4,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return sum(out) / len(out)


def test_a_bare_invocation_explains_itself_and_fails() -> None:
    """Exit 2, not 0.

    Forgetting the verb is a caller error, and a shell script checking the exit
    code should be told about it.
    """
    assert main([]) == 2


def test_help_is_not_an_error() -> None:
    assert main(["--help"]) == 0
    assert main(["help"]) == 0


def test_an_unknown_verb_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["wat"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_frame_writes_the_image_and_prints_its_path(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "f.jpg"
    assert main(["frame", str(video), "1.0", "-o", str(dest)]) == 0
    assert dest.exists()
    # The path alone on stdout, so a caller can use it directly.
    assert capsys.readouterr().out.strip() == str(dest)


def test_frame_seeks_to_the_second_asked_for(video: Path, tmp_path: Path) -> None:
    """The fixture is near-black for 0-4s and near-white after.

    A seek that landed in the wrong half is invisible to "a file was written".
    """
    early = tmp_path / "early.png"
    late = tmp_path / "late.png"
    main(["frame", str(video), "1.0", "-o", str(early)])
    main(["frame", str(video), "6.0", "-o", str(late)])
    assert _mean_grey(early) < 64 < 192 < _mean_grey(late)


def test_frame_scales_to_the_requested_width(video: Path, tmp_path: Path) -> None:
    """Aspect preserved, height even.

    `scale=W:-2`, not `-1`: the latter can produce an odd height that some
    encoders refuse outright.
    """
    dest = tmp_path / "f.jpg"
    main(["frame", str(video), "1.0", "-o", str(dest), "--width", "160"])
    assert _dimensions(dest) == (160, 120)


def test_width_zero_keeps_the_source_resolution(video: Path, tmp_path: Path) -> None:
    dest = tmp_path / "f.jpg"
    main(["frame", str(video), "1.0", "-o", str(dest), "--width", "0"])
    assert _dimensions(dest) == (320, 240)


def test_a_bad_timestamp_surfaces_rather_than_writing_nothing(
    video: Path, tmp_path: Path
) -> None:
    from deixis.media import MediaError

    with pytest.raises(MediaError, match="past the end"):
        main(["frame", str(video), "9999", "-o", str(tmp_path / "f.jpg")])


def test_mark_routes_to_the_frames_cli(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 0

    monkeypatch.setattr("deixis.frames.main", fake)
    assert main(["mark", str(video), "-t", "t.json"]) == 0
    assert seen == [[str(video), "-t", "t.json"]]


def test_transcribe_routes_to_the_transcribe_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 0

    monkeypatch.setattr("deixis.transcribe.main", fake)
    assert main(["transcribe", "v.mov", "-o", "out.json"]) == 0
    assert seen == [["v.mov", "-o", "out.json"]]


def test_the_installed_console_script_works() -> None:
    """The entry point itself, not just the function behind it.

    A `[project.scripts]` typo is invisible to every test that imports main()
    directly -- and the console script is the whole point of this file.
    """
    proc = subprocess.run(
        ["uv", "run", "deixis", "--help"], capture_output=True, text=True, check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr
    assert "frame      VIDEO SECONDS" in proc.stdout
