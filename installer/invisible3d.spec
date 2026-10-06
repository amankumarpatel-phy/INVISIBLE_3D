# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# PyInstaller evaluates paths in a spec relative to the spec file's directory.
# Resolve the repository root explicitly so this file works from GitHub Actions
# and from a local checkout.
ROOT = Path.cwd()

hiddenimports = collect_submodules("INVISIBLE_3D")
datas = collect_data_files("INVISIBLE_3D")

a = Analysis(
    [str(ROOT / "desktop_app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="INVISIBLE3D",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="INVISIBLE3D",
)
