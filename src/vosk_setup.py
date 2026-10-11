"""One-time, atomic cached download of the small free English Vosk model."""
from __future__ import annotations

import argparse
import platform
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .vosk_wake import DEFAULT_MODEL_URL, VoskWakeWordError, model_path


def ensure_model(*, url: str = DEFAULT_MODEL_URL, target: Path | None = None) -> Path:
    target = target or model_path()
    if (target / "am" / "final.mdl").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ziggy-vosk-", dir=target.parent) as temporary:
        archive = Path(temporary) / "model.zip"
        try:
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as contents:
                root = Path(temporary).resolve()
                for member in contents.infolist():
                    destination = (root / member.filename).resolve()
                    if root not in destination.parents and destination != root:
                        raise VoskWakeWordError("Vosk archive contains an unsafe path")
                contents.extractall(temporary)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise VoskWakeWordError(f"Unable to download/extract the Vosk model from {url}: {exc}") from exc
        extracted = next((path for path in Path(temporary).iterdir()
                          if path.is_dir() and (path / "am" / "final.mdl").is_file()), None)
        if extracted is None:
            raise VoskWakeWordError("Downloaded Vosk archive did not contain a valid small English model")
        if target.exists():
            raise VoskWakeWordError(
                f"Model target {target} exists but is not a valid Vosk model; remove or rename it before setup")
        extracted.replace(target)
    return target


def verify_vosk() -> None:
    try:
        import vosk
        _ = vosk
    except ImportError as exc:
        raise VoskWakeWordError(
            f"Vosk did not install for Python {platform.python_version()} on {platform.machine()}. "
            "Use a Raspberry Pi OS Python version supported by the Vosk wheel, then rerun pip install -r requirements.txt.") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install/cache Ziggy's free offline Vosk wake model")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    args = parser.parse_args(argv)
    verify_vosk()
    path = ensure_model(url=args.model_url)
    print(f"Ziggy Vosk model ready: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VoskWakeWordError as exc:
        print(f"Vosk setup error: {exc}")
        raise SystemExit(2)
