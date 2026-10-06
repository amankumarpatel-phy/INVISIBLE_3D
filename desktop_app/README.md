# INVISIBLE³D Desktop

The desktop edition is a standalone Windows GUI around the INVISIBLE³D
computational imaging engine.

## Run from source

Install the desktop dependencies:

    pip install -r requirements.txt

Then:

    python desktop_app/main.py

## Build

The Windows installer is built automatically by GitHub Actions using
PyInstaller and Inno Setup. The output is a single installable Setup.exe
artifact/release asset.

The installed application does not require Python to be installed on the
target Windows PC.
