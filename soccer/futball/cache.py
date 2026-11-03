"""Small bounded caches invalidated by file replacement and calendar rollover."""
from collections import OrderedDict
from datetime import date
from functools import wraps
from pathlib import Path
from threading import RLock


def _today():
    return date.today().isoformat()


def _version(path):
    path = Path(path)
    try:
        stat = path.stat()
        return str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return str(path), None


def file_cache(*paths, daily=False, maxsize=1):
    """Paths may be callables so runtime configuration remains testable.

    Producers should replace files atomically. Cache misses are serialized to
    prevent concurrent requests from duplicating feature/model loading work.
    """
    def decorate(fn):
        cache = OrderedDict()
        lock = RLock()
        @wraps(fn)
        def wrapped(*args, **kwargs):
            versions = tuple(_version(p() if callable(p) else p) for p in paths)
            key = (versions, _today() if daily else None, args, tuple(sorted(kwargs.items())))
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
                value = fn(*args, **kwargs)
                cache[key] = value
                while len(cache) > maxsize:
                    cache.popitem(last=False)
                return value
        def clear():
            with lock:
                cache.clear()
        wrapped.cache_clear = clear
        return wrapped
    return decorate
