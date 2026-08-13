"""The install path the README publishes, executed rather than described.

Every other test in this suite runs against the working tree through `uv run`,
where `.python-version` pins the interpreter to 3.12 and the extras are already
synced. That is not how anyone installs dsj. The README tells a stranger to
run `uv tool install`, which has no project context, reads no
`.python-version`, and resolves its own interpreter from `requires-python`.

Those two paths diverged and nobody noticed, because nothing executed the
second one. `uv tool install` picked Python 3.14; coremltools publishes no
wheel above 3.13 and declares no `requires-python`, so it built from source,
installed cleanly, and failed to load hours later on a 45-minute recording.
The suite was green throughout.

So this gate runs the documented command itself, with the extra, and imports
the thing that broke.

WHY THERE IS NO `skipif` FOR A MISSING senko HERE. tests/test_diarize_gate.py
skips when the diarize extra is absent, which is correct for a gate about
diarization quality and is exactly the shape that hid this bug: a check that
switches itself off in the state it exists to detect. senko being unimportable
is this gate's failure condition, not its skip condition.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# The literal line the README publishes, minus the remote. Kept as a pattern
# rather than a copy so that editing the README's command and not this file is
# a test failure rather than a silent drift.
DOCUMENTED = re.compile(
    r'uv tool install "dsj\[diarize\] @ git\+https://github\.com/m2moiz/dekho-suno-jaano"'
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH"),
]


def _uv(*args: str, tool_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run uv with an isolated tool directory.

    UV_TOOL_DIR and UV_TOOL_BIN_DIR are what keep this from touching the
    operator's real `uv tool` installs -- the gate must not be able to clobber
    a working dsj on the machine running it.
    """
    env = os.environ | {
        "UV_TOOL_DIR": str(tool_dir),
        "UV_TOOL_BIN_DIR": str(tool_dir / "bin"),
    }
    return subprocess.run(
        ["uv", *args], capture_output=True, text=True, env=env, check=False, timeout=900
    )


def test_the_readme_still_publishes_the_command_this_gate_runs() -> None:
    """The gate is only worth anything while it tests the published command.

    A cheap fast check that the two have not drifted. If the README's install
    line changes, this fails and points at the gate below rather than letting
    it go on verifying a command nobody is told to run.
    """
    assert DOCUMENTED.search(README.read_text()), (
        "the README no longer publishes the install command this gate runs; "
        "update DOCUMENTED and the gate together"
    )


def test_the_documented_install_produces_a_working_diarizing_dsj(
    tmp_path: Path,
) -> None:
    """Run the README's command, then import the extra that broke.

    The only substitution is the remote: a local `git+file://` clone stands in
    for github.com, so the gate tests the branch under test rather than
    whatever is on main. Everything else -- the extra, the resolver, the
    interpreter choice -- is the published path untouched.
    """
    tool_dir = tmp_path / "tools"
    spec = f'dsj[diarize] @ git+file://{REPO}'

    install = _uv("tool", "install", "--force", spec, tool_dir=tool_dir)
    assert install.returncode == 0, (
        f"the documented install failed:\n{install.stdout}\n{install.stderr}"
    )

    # 1. The console script exists. `uv sync` puts it somewhere nothing can
    #    reach; this is the difference the README's install section is about.
    binary = tool_dir / "bin" / "dsj"
    assert binary.is_file(), f"no dsj executable in {tool_dir / 'bin'}"

    # 2. It runs.
    helped = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, check=False, timeout=120
    )
    assert helped.returncode == 0, helped.stderr
    assert "dikhao" in helped.stdout

    env_python = tool_dir / "dsj" / "bin" / "python"
    assert env_python.is_file(), f"no interpreter in the tool env at {env_python}"

    # 3. The interpreter obeys the cap. This is the bug itself: with
    #    `requires-python = ">=3.12"` uv picked 3.14, where the next assertion
    #    fails. Checked separately so a future regression says WHICH half broke.
    version = subprocess.run(
        [str(env_python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout.strip()
    major, minor = (int(part) for part in version.split("."))
    assert (major, minor) >= (3, 12), f"tool env is on Python {version}, below the floor"
    assert (major, minor) < (3, 14), (
        f"tool env is on Python {version}. requires-python let uv pick an "
        f"interpreter above the cap, which is how coremltools ends up built "
        f"from source and unloadable."
    )

    # 4. senko imports. Necessary, and NOT sufficient -- see the next
    #    assertion, which is the one this file exists for.
    imported = subprocess.run(
        [str(env_python), "-c", "import senko; print(senko.__name__)"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert imported.returncode == 0, (
        f"dsj[diarize] installed on Python {version} and senko will not "
        f"import:\n{imported.stderr}"
    )

    # 5. THE ASSERTION THIS FILE EXISTS FOR: coremltools brought its native
    #    extensions.
    #
    #    Measured on a forced 3.14 install: coremltools publishes no wheel for
    #    that interpreter, so uv builds the sdist -- and the build SUCCEEDS,
    #    producing a `py3-none-any` wheel with the CoreML extensions simply
    #    absent. Nothing raises. `import coremltools` prints "Failed to load
    #    '_MLGPUComputeDeviceRemoteProxy'" to stderr and carries on; `import
    #    senko` is clean. The install is green, the import is green, and
    #    diarization then does not work -- which dsj's fail-soft boundary
    #    turns into a transcript quietly missing its speaker labels.
    #
    #    So an import check cannot catch this and a returncode cannot either.
    #    Root-Is-Purelib is the signal: a real coremltools ships a platform
    #    wheel, and a gutted one does not.
    wheel_tag = subprocess.run(
        [
            str(env_python),
            "-c",
            "import importlib.metadata as m\n"
            "d = m.distribution('coremltools')\n"
            "w = d.read_text('WHEEL') or ''\n"
            "print([ln for ln in w.splitlines() if ln.startswith('Tag:')] or ['Tag: <none>'])",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert wheel_tag.returncode == 0, wheel_tag.stderr
    assert "py3-none-any" not in wheel_tag.stdout, (
        f"coremltools installed as a pure-Python wheel {wheel_tag.stdout.strip()} on "
        f"Python {version}: the native CoreML extensions were not built. It will "
        f"import cleanly and diarization will not run."
    )

    # 5. And dsj's own boundary agrees. `_import_senko` is what converts a
    #    load failure into DiarizationUnavailable, and a green import above
    #    with a raising boundary here would mean the boundary is lying.
    boundary = subprocess.run(
        [
            str(env_python),
            "-c",
            "import json\n"
            "from dsj.diarize import _import_senko\n"
            "print(json.dumps({'module': _import_senko().__name__}))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert boundary.returncode == 0, (
        f"senko imports but dsj's diarize boundary still refuses it:\n{boundary.stderr}"
    )
    assert json.loads(boundary.stdout)["module"] == "senko"


def test_this_gate_runs_on_a_python_the_cap_allows() -> None:
    """A guard on the guard.

    The assertions above compare the TOOL ENV's interpreter against the cap.
    If the suite itself is ever run on a Python outside the supported range,
    that comparison is still valid but the rest of the suite is not, and it
    should say so once rather than fail obscurely everywhere.
    """
    assert (3, 12) <= sys.version_info[:2] < (3, 14), (
        f"the suite is running on Python {sys.version_info.major}."
        f"{sys.version_info.minor}, outside pyproject's requires-python"
    )
