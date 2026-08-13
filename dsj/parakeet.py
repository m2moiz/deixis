"""The parakeet engine: the code that may import parakeet-mlx, and nobody else.

Extracted from suno.py and chunking.py so the default engine finally gets the
treatment whisper.py and diarize.py always had -- one backend, one leaf module,
lazy imports behind an availability probe. The AST test in
tests/test_core_is_portable.py is what holds this line: no core module may
import parakeet_mlx, mlx, or any other backend, ever again.

Module top is backend-free ON PURPOSE. The registry imports this module before
calling available(), so a top-level `import parakeet_mlx` would crash the
resolution this module exists to make safe. Every backend import lives inside
the function that needs it, which is also what lets the test suite patch the
source attributes (parakeet_mlx.from_pretrained and friends) and have the
function-local imports pick the patches up.
"""

from __future__ import annotations

__all__ = ["DEFAULT_MODEL", "available", "fingerprint_fields", "load", "wrap"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from dsj.alignment import AlignedToken

if TYPE_CHECKING:
    from pathlib import Path

    from parakeet_mlx.alignment import AlignedResult as UpstreamResult

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def available() -> str | None:
    """None if the backend would import; otherwise the reason it will not.

    find_spec, never an import: this runs adjacent to `--help` and on machines
    where the import would fail, and both must stay cheap and safe.
    """
    import sys
    from importlib.util import find_spec

    # sys.modules first, for the same reason as dsj.whisper.available: an
    # imported (or test-stubbed) backend is available, and find_spec raises
    # on a stub with no __spec__.
    if "parakeet_mlx" in sys.modules:
        return None
    if find_spec("parakeet_mlx") is None:
        import platform

        return (
            "parakeet-mlx is not installed. It requires Apple Silicon and "
            f"Metal; this machine is {platform.machine()} {platform.system()}. "
            "On a Mac, reinstall with the parakeet engine included (see the "
            "README's install line)."
        )
    return None


def fingerprint_fields() -> dict[str, str]:
    """The engine's contribution to a checkpoint fingerprint, merged flat.

    The key NAME is load-bearing: `parakeet_version` is what every checkpoint
    on disk already carries from when checkpoint.py hardcoded it, and the flat
    merge in Fingerprint.to_dict() only reproduces those bytes if this key
    matches. The version matters for the reason the old comment gave -- the
    merge helpers were vendored from parakeet-mlx, and tokens merged by one
    version's functions cannot safely be extended by another's.
    """
    from importlib.metadata import version

    return {"parakeet_version": version("parakeet-mlx")}


class _Generates(Protocol):
    """The one method this engine calls on a loaded model.

    BaseParakeet.generate is annotated upstream, but its `mel: mx.array`
    parameter resolves to Unknown (mlx ships no stubs), which makes every call
    through it partially unknown. Restating the signature with the mel as Any
    keeps the useful half -- the list[AlignedResult] return -- typed.
    """

    def generate(
        self, mel: Any, *, decoding_config: Any = ...
    ) -> list[UpstreamResult]: ...


@dataclass
class _LoadedParakeet:
    """A loaded model behind the ChunkEngine protocol (dsj.asr).

    Holds the model handle and the decoding config; the feature step
    (get_logmel) is re-imported per decode so a monkeypatched
    parakeet_mlx.audio is honoured -- the same reason every other backend
    import here is function-local.
    """

    model: Any
    sample_rate: int
    min_chunk_samples: int
    _decoding: Any = field(default=None, repr=False)

    def load_audio(self, path: Path) -> Any:
        """Decode `path` to the model's sample rate. Returns an mx.array."""
        from parakeet_mlx.audio import (
            load_audio as _load_audio,  # pyright: ignore[reportUnknownVariableType]  # mlx has no stubs
        )

        return cast("Any", _load_audio)(path, self.sample_rate)

    def decode(self, samples: Any) -> list[AlignedToken]:
        """One chunk's tokens, timed from the chunk's own start (t=0)."""
        from parakeet_mlx.audio import (
            get_logmel as _get_logmel,  # pyright: ignore[reportUnknownVariableType]  # mlx has no stubs
        )

        mel = cast("Any", _get_logmel)(samples, self.model.preprocessor_config)
        result = cast(_Generates, self.model).generate(
            mel, decoding_config=self._decoding
        )[0]
        # The vocabulary boundary: upstream tokens become dsj tokens here, and
        # nothing above this module ever sees a parakeet_mlx type again.
        return [
            AlignedToken(
                id=token.id,
                text=token.text,
                start=token.start,
                duration=token.duration,
                confidence=token.confidence,
            )
            for sentence in result.sentences
            for token in sentence.tokens
        ]


def wrap(model: Any) -> _LoadedParakeet:
    """Wrap an already-loaded model (or a test double) as a ChunkEngine.

    Split from load() so the slow equivalence tests, which hold a real model,
    and the fast tests, which hold a FakeModel, drive the exact engine object
    production uses rather than a lookalike.
    """
    from parakeet_mlx import DecodingConfig

    return _LoadedParakeet(
        model=model,
        # A property of this model's preprocessor, not a constant we assume.
        sample_rate=model.preprocessor_config.sample_rate,
        # Below one hop there is no feature frame to decode -- upstream's own
        # guard, restated as a number chunking.py can compare against.
        min_chunk_samples=model.preprocessor_config.hop_length,
        _decoding=DecodingConfig(),
    )


def load(model_id: str = DEFAULT_MODEL) -> _LoadedParakeet:
    """Download/load the weights and return the engine. The expensive call."""
    from parakeet_mlx import (
        from_pretrained as _from_pretrained,  # pyright: ignore[reportUnknownVariableType]  # mlx has no stubs
    )

    return wrap(cast("Any", _from_pretrained)(model_id))
