"""Optional external artifact and experiment tracking adapters."""


from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def upload(self, local_path: Path, remote_path: str) -> None: ...

    def download(self, remote_path: str, local_path: Path) -> None: ...


class LocalArtifactStore:
    """Minimal local adapter useful for tests and development."""

    def upload(self, local_path: Path, remote_path: str) -> None:
        destination = Path(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(local_path.read_bytes())

    def download(self, remote_path: str, local_path: Path) -> None:
        source = Path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(source.read_bytes())

