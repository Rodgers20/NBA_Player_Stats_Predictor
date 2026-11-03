"""Filesystem plumbing: atomic publication, and listing that survives exFAT."""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import os
import tempfile

# The data volume is exFAT, which has no native extended attributes. macOS
# stores them in an AppleDouble companion instead — `._E0.csv` beside `E0.csv` —
# and those companions match every glob the data layer uses while containing
# binary AppleDouble headers rather than data. Reading one raises
# UnicodeDecodeError; parsing its name raises ValueError. They are invisible in
# Finder and in `ls` without -a, which makes the resulting crash baffling.
_SIDECAR_PREFIX = "._"
_MACOS_METADATA = frozenset({
    ".DS_Store", ".Spotlight-V100", ".Trashes", ".fseventsd",
    ".TemporaryItems", ".DocumentRevisions-V100",
})


def is_sidecar(path) -> bool:
    """True for a file the operating system wrote, not one we did."""
    name = Path(path).name
    if name.startswith(_SIDECAR_PREFIX):
        return True
    if name in _MACOS_METADATA:
        return True
    # atomic_path's scratch files, left behind only if a process was killed.
    return name.startswith(".") and name.endswith(".partial")


def data_files(directory, pattern: str, *, non_empty: bool = False) -> Iterator[Path]:
    """Sorted real data files under `directory` matching `pattern`.

    Use this instead of `Path.glob` anywhere the target lives on the data
    volume. Every path component is checked, so a pattern spanning directories
    (`*/E0.csv`) skips a sidecar-named parent too, and directories that happen to
    match the pattern are excluded so callers can assume they have a file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob(pattern)):
        if any(is_sidecar(part) for part in path.relative_to(directory).parts):
            continue
        if not path.is_file():
            continue
        if non_empty and path.stat().st_size == 0:
            continue
        yield path


@contextmanager
def atomic_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.partial', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(destination, text):
    with atomic_path(destination) as temporary:
        temporary.write_text(text, encoding='utf-8')
