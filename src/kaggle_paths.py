"""Locate inputs on Kaggle, which mounts competitions either at /kaggle/input/<slug> or
/kaggle/input/competitions/<slug>, and datasets at /kaggle/input/<slug> or deeper."""
import glob
import os
from typing import Optional

COMPETITION = "rsna-knee-abnormality-detection"


def find_competition_root(root: str = "/kaggle/input") -> Optional[str]:
    """Directory holding test.csv + test_series/ for this competition."""
    for pat in (f"{root}/{COMPETITION}", f"{root}/competitions/{COMPETITION}", f"{root}/*", f"{root}/*/*"):
        for d in sorted(glob.glob(pat)):
            if os.path.isfile(os.path.join(d, "test.csv")) and os.path.isdir(os.path.join(d, "test_series")):
                return d
    return None


def find_file(name: str, root: str = "/kaggle/input", max_depth: int = 5) -> Optional[str]:
    """First file called `name` under root (depth-limited; skips the huge DICOM trees)."""
    root = root.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if not d.endswith("_series")]
        if name in filenames:
            return os.path.join(dirpath, name)
    return None
