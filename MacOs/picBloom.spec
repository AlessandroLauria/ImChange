# -*- mode: python -*-

block_cipher = None


a = Analysis(['picBloom.py'],
             pathex=['/Users/alessandrolauria/Desktop/picBloom'],
             binaries=[],
             datas=[('Images/*.png','Images')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='picBloom',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='picBloom')
app = BUNDLE(coll,
             name='picBloom.app',
             icon='Images/icon.icns',
             bundle_identifier=None)
