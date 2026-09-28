from setuptools import find_packages, setup

setup(
    name="rsna_knee",
    version="0.1.0",
    description="RSNA Knee Abnormality Detection 2026 — training & inference pipeline",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
