#!/usr/bin/env python3
"""
Offline Asset Setup Script for Quantum Operator Algebra Calculator.

This script downloads KaTeX (CSS, JS, auto-render extension, and fonts) into
the local `static/katex/` directory so the application can run in a completely
air-gapped / offline environment without internet access.

Usage:
    python setup_offline.py
    python setup_offline.py --wheels    (also downloads pip wheels for offline Python install)
"""

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

KATEX_VERSION = "0.16.11"
GITHUB_ZIP_URL = f"https://github.com/KaTeX/KaTeX/releases/download/v{KATEX_VERSION}/katex.zip"
NPM_TGZ_URL = f"https://registry.npmjs.org/katex/-/katex-{KATEX_VERSION}.tgz"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
KATEX_DIR = STATIC_DIR / "katex"


def download_stream(url: str, timeout: int = 30) -> bytes:
    """Download data from URL with a standard User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else 0
        buf = io.BytesIO()
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)
            if total_bytes > 0:
                percent = downloaded / total_bytes * 100
                print(f"\rDownloading: {downloaded / 1024 / 1024:.2f} MB / {total_bytes / 1024 / 1024:.2f} MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloading: {downloaded / 1024 / 1024:.2f} MB", end="", flush=True)
        print()
        return buf.getvalue()


def install_katex():
    print(f"=== Setting up offline KaTeX v{KATEX_VERSION} ===")
    KATEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Try GitHub Release ZIP first
    download_ok = False
    print(f"Attempting download from GitHub release: {GITHUB_ZIP_URL}")
    try:
        data = download_stream(GITHUB_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            prefix = "katex/"
            for member in zf.infolist():
                if member.filename.startswith(prefix):
                    rel_path = member.filename[len(prefix):]
                    if not rel_path:
                        continue
                    dest = KATEX_DIR / rel_path
                    if member.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        download_ok = True
        print(f"[OK] Successfully extracted KaTeX distribution from GitHub zip.")
    except Exception as e:
        print(f"[WARN] GitHub download failed ({e}), trying NPM fallback...")

    # 2. Try NPM tarball as fallback
    if not download_ok:
        print(f"Attempting download from NPM registry: {NPM_TGZ_URL}")
        try:
            data = download_stream(NPM_TGZ_URL)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                dist_prefix = "package/dist/"
                for member in tf.getmembers():
                    if member.name.startswith(dist_prefix):
                        rel_path = member.name[len(dist_prefix):]
                        if not rel_path:
                            continue
                        dest = KATEX_DIR / rel_path
                        if member.isdir():
                            dest.mkdir(parents=True, exist_ok=True)
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            f = tf.extractfile(member)
                            if f:
                                with open(dest, "wb") as dst:
                                    shutil.copyfileobj(f, dst)
            download_ok = True
            print(f"[OK] Successfully extracted KaTeX distribution from NPM tarball.")
        except Exception as e:
            print(f"[ERROR] NPM fallback download also failed: {e}")
            sys.exit(1)

    # Verify essential assets
    required_files = [
        KATEX_DIR / "katex.min.css",
        KATEX_DIR / "katex.min.js",
        KATEX_DIR / "contrib" / "auto-render.min.js",
        KATEX_DIR / "fonts",
    ]
    all_ok = True
    for p in required_files:
        if not p.exists():
            print(f"[MISSING] Required asset: {p}")
            all_ok = False
        else:
            print(f"  + Verified: {p.relative_to(BASE_DIR)}")

    if all_ok:
        print("\n[SUCCESS] KaTeX offline assets are ready in ./static/katex/")
    else:
        print("\n[WARNING] Some assets appear to be missing. Please verify the setup.")


def download_wheels():
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        print(f"[SKIP] No requirements.txt found in {BASE_DIR}")
        return

    wheels_dir = BASE_DIR / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    print(f"\n=== Downloading Python dependency wheels to {wheels_dir} ===")
    cmd = [sys.executable, "-m", "pip", "download", "-r", str(req_file), "-d", str(wheels_dir)]
    try:
        subprocess.check_call(cmd)
        print(f"[SUCCESS] Python wheels downloaded. To install offline:")
        print(f"  pip install --no-index --find-links=./wheels -r requirements.txt")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download wheels: {e}")


def main():
    install_katex()

    if "--wheels" in sys.argv:
        download_wheels()
    else:
        print("\nTip: Run `python setup_offline.py --wheels` to also download all Python pip wheels for offline installation.")


if __name__ == "__main__":
    main()

