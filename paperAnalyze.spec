# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['paperAnalyze.py'],
             pathex=['E:\\SRTP文献检索系统\\提交代码_by 何颖智\\paperAnaylze.exe Python源码'],
             binaries=[],
             datas=[],
             hiddenimports=['numpy.core._dtype_ctypes'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='paperAnalyze',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
