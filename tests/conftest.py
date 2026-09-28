"""Synthetic knee-competition fixture: tiny DICOM studies + csvs in the real layout."""
import os

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.constants import ID_COL, LABELS

PLANES = {  # ImageOrientationPatient per plane
    "Sagittal": [0, 1, 0, 0, 0, -1],
    "Coronal": [1, 0, 0, 0, 0, -1],
    "Axial": [1, 0, 0, 0, 1, 0],
}


def write_slice(path, pixels, plane, z, instance):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    iop = PLANES[plane]
    normal = np.cross(iop[:3], iop[3:])
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = list((normal * z).astype(float))
    ds.InstanceNumber = instance
    ds.PixelSpacing = [0.5, 0.5]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def make_study(root, uid, rng, n_slices=12, size=64):
    rows = []
    for plane in ("Sagittal", "Coronal", "Axial"):
        for fluid in (1, 0):
            suid = generate_uid()
            d = os.path.join(root, uid, suid)
            os.makedirs(d)
            order = rng.permutation(n_slices)  # file names / instance numbers deliberately shuffled
            for i, z in enumerate(order):
                px = np.full((size, size), 100 + 50 * z, dtype=np.uint16)  # intensity encodes true position
                write_slice(os.path.join(d, f"{generate_uid()}.dcm"), px, plane, float(z), int(i))
            rows.append({ID_COL: uid, "SeriesInstanceUID": suid, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    return rows


@pytest.fixture(scope="session")
def fake_comp(tmp_path_factory):
    """data_dir with raw/{train,test}.csv, *_series.csv, *_series/ DICOMs and labels/*.csv."""
    root = tmp_path_factory.mktemp("comp")
    raw, lab = root / "raw", root / "labels"
    raw.mkdir(), lab.mkdir()
    rng = np.random.default_rng(0)
    train_ids = [generate_uid() for _ in range(14)]
    test_ids = [generate_uid() for _ in range(3)]
    series = sum((make_study(str(raw / "train_series"), u, rng) for u in train_ids), [])
    tseries = sum((make_study(str(raw / "test_series"), u, rng) for u in test_ids), [])
    pd.DataFrame(series).to_csv(raw / "train_series.csv", index=False)
    pd.DataFrame(tseries).to_csv(raw / "test_series.csv", index=False)

    y = (rng.random((len(train_ids), len(LABELS))) > 0.5).astype(float)
    y[:2] = [[1, 0] * 6, [0, 1] * 6]  # guarantees both classes per column in any split
    train = pd.DataFrame({ID_COL: train_ids, "Report": "no report"})
    gold = pd.DataFrame(np.nan, index=range(len(train_ids)), columns=LABELS)
    gold.iloc[-4:, :] = np.vstack([[1, 0] * 6, [0, 1] * 6, [1, 0] * 6, [0, 1] * 6])  # last 4 studies play the "gold" role
    pd.concat([train, gold], axis=1).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({ID_COL: test_ids}).to_csv(raw / "test.csv", index=False)
    lab_df = pd.DataFrame(y, columns=LABELS)
    lab_df.insert(0, ID_COL, train_ids)
    lab_df.iloc[0, 1] = 0.5  # one "not addressed" cell
    lab_df.to_csv(lab / "fake_labels.csv", index=False)
    return root
