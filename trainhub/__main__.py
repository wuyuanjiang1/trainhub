import sys

from .app import _fix_null_streams
from .app import run

# Spawn-based multiprocessing children re-execute this module with
# __name__ == "__mp_main__", so the guard must sit above the entry check.
_fix_null_streams()

if __name__ == "__main__":
    sys.exit(run())
