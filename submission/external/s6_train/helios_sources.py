"""hlly34: lay out heliosli's code the way s6common expects it, without copying it into the repo twice.

  <dest>/            the six sources (px_cache, compact_train, compact_infer, probe_infer, knee_anchor_bank_v2, side_common),
                     taken from the SOURCES dict of ../heliosli_compact_stage.py (heliosli's public notebook, verbatim) and
                     checked against s6common.SOURCE_SHA (= nartaa's heliosli.SHA256SUMS)
  <dest>/cloud_code  heliosli's data pipeline from the dataset heliosli/rsna-orthofoundation-fold0-epoch7-20261004
                     (copied, or unzipped from cloud_code.zip)

  python helios_sources.py --dest DIR --ortho /kaggle/input/.../rsna-orthofoundation-fold0-epoch7-20261004 [--wheels]
--wheels also pip-installs the dataset's pinned wheels (MONAI etc.), as the inference stage does."""
import argparse, ast, hashlib, os, shutil, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s6common as SC  # noqa: E402

STAGE = HERE.parent / "heliosli_compact_stage.py"


def sources(stage=STAGE):
    """The SOURCES dict literal of the verbatim stage file, parsed (not executed)."""
    tree = ast.parse(Path(stage).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SOURCES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("no SOURCES dict in " + str(STAGE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--ortho", required=True, help="mount of heliosli/rsna-orthofoundation-fold0-epoch7-20261004")
    ap.add_argument("--wheels", action="store_true")
    ap.add_argument("--stage", default=str(STAGE), help="heliosli_compact_stage.py (default: next to this directory)")
    a = ap.parse_args()
    dest, ortho = Path(a.dest), Path(a.ortho)
    dest.mkdir(parents=True, exist_ok=True)
    src = sources(a.stage)
    for name, want in SC.SOURCE_SHA.items():
        data = src[name].encode("utf-8")
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit("%s differs from SOURCE_SHA (%s != %s)" % (name, got[:12], want[:12]))
        (dest / name).write_bytes(data)
    code = dest / "cloud_code"
    if not code.exists():
        if (ortho / "cloud_code.zip").is_file():
            with zipfile.ZipFile(ortho / "cloud_code.zip") as z:
                z.extractall(code)
        else:
            shutil.copytree(ortho / "cloud_code", code)
    if a.wheels:
        wheels = sorted(str(p) for p in ortho.rglob("*.whl"))
        if (ortho / "dependencies.zip").is_file():
            wd = dest / "_wheels"
            wd.mkdir(exist_ok=True)
            with zipfile.ZipFile(ortho / "dependencies.zip") as z:
                for n in z.namelist():
                    if n.endswith(".whl"):
                        (wd / Path(n).name).write_bytes(z.read(n))
            wheels += sorted(str(p) for p in wd.glob("*.whl"))
        if not wheels:
            raise SystemExit("no wheels in " + str(ortho))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=True)
    print("heliosli sources ->", dest, "| cloud_code ->", code, "| anatomy", ortho / "anatomy.pt", flush=True)


if __name__ == "__main__":
    main()
