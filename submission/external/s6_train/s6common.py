"""S6F0 shared paths and process setup (imported by every S6F0 script, and by every DataLoader worker through them).

Heliosli's code is imported ONLY from the read-only, sha-checked copies in S6F0/heliosli/ (sources) and
S6F0/heliosli/cloud_code/ (the data pipeline), never from helios954/.  heliosli.SHA256SUMS / cloud_code.SHA256SUMS hold the
hashes; verify_copies() re-checks the six sources against the 0.954 notebook's SOURCE_SHA at start-up."""
import hashlib, json, os, sys, types
from datetime import datetime, timedelta, timezone

S6F0 = os.path.dirname(os.path.abspath(__file__))
# hlly34: nartaa's machine paths became env overrides (defaults = their layout); helios_sources.py writes heliosli/
HELIOS_SRC = os.environ.get("S6_HELIOS_SRC", os.path.join(S6F0, "heliosli"))
CODE = os.environ.get("S6_CODE", os.path.join(HELIOS_SRC, "cloud_code"))
ANATOMY = os.environ.get("S6_ANATOMY", "/home/danial/rsna-knee/research/gap-hunt-20261008/exec/helios954/orthofoundation_20261004/anatomy.pt")
ANATOMY_SHA256 = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"     # S6 manifest anchor_weights_sha256
DATA = os.environ.get("S6_DATA", "/home/danial/rsna-knee/raw")
SPLIT = "train"
ANCHORS = os.environ.get("S6_ANCHORS", os.path.join(S6F0, "anchors"))
DATA_DIR = os.path.join(S6F0, "data")
# the 0.954 notebook's SOURCE_SHA (helios954/src/rsna-knee-abnormality-detection.ipynb); S6 manifest src_sha256 agrees
SOURCE_SHA = {
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
    "compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
}
AMMAN = timezone(timedelta(hours=3))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_copies(anatomy=False):
    for n, want in SOURCE_SHA.items():
        got = sha256_file(os.path.join(HELIOS_SRC, n))
        if got != want:
            raise SystemExit("S6F0/heliosli/%s differs from the notebook SOURCE_SHA (%s != %s)" % (n, got[:12], want[:12]))
    if anatomy and sha256_file(ANATOMY) != ANATOMY_SHA256:
        raise SystemExit("anatomy.pt differs from the S6 manifest anchor checkpoint")


def install_paths():
    """sys.path = [cloud_code, heliosli copies, S6F0]; stub training.download_data (it imports `requests`, absent in .venv312;
    only the chunked-archive reader uses it, and we read the DICOM directory).  Idempotent; safe in spawn and fork workers."""
    sys.dont_write_bytecode = True
    for p in (S6F0, HELIOS_SRC, CODE):          # inserted in reverse: CODE ends up first, as compact_infer does (code, HERE)
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data")
        m.ChunkedArchive = None
        sys.modules["training.download_data"] = m


def lean_native():
    """probe_infer.install_native(True) -- the exact call compact_infer.CStudies._setup makes -- once per process.
    It is NOT idempotent (a second call re-reads the already patched source and raises 'differs from the frozen source'),
    and a fork child inherits the patched module, so guard it with a module attribute."""
    install_paths()
    import training.native_cache as native
    if not getattr(native, "_s6f0_lean_installed", False):
        import probe_infer as PI
        PI.install_native(True)
        native._s6f0_lean_installed = True
    return native


def study_key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:24]


def anchor_path(uid, anchors=ANCHORS):
    return os.path.join(anchors, study_key(uid) + ".json")


def amman_now():
    return datetime.now(AMMAN)


def amman(ts=None):
    return (datetime.now(AMMAN) if ts is None else datetime.fromtimestamp(ts, AMMAN)).strftime("%a %H:%M")


def atomic_json(obj, path, **kw):
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_json(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


def fsync_dir(d):
    """S6F0 review fix 10-10: make a rename in directory d durable (checkpoint / model writes)."""
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
