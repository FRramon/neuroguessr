# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['neuroguessr.py'],
    pathex=[],
    binaries=[],
    datas=[('../data', 'data'), ('../code', 'code')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroGuessr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['neuroguessr5.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuroGuessr',
)
app = BUNDLE(
    coll,
    name='NeuroGuessr.app',
    icon='neuroguessr5.icns',
    bundle_identifier=None,
)
