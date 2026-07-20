#!/usr/bin/env python
# Debug ImageMagick setup

import shutil
import subprocess

print("=" * 60)
print("Checking ImageMagick Setup")
print("=" * 60)

# Check if magick is in PATH
print("\n1. Checking if magick is in system PATH:")
which_magick = shutil.which("magick")
print(f"   shutil.which('magick'): {which_magick}")

# Check if convert is in PATH (older ImageMagick)
print("\n2. Checking if convert is in system PATH (legacy):")
which_convert = shutil.which("convert")
print(f"   shutil.which('convert'): {which_convert}")

# Try to run magick --version
print("\n3. Testing magick --version:")
try:
    result = subprocess.run(
        ["magick", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"   Return code: {result.returncode}")
    if result.returncode == 0:
        print(f"   ✅ magick works!")
        print(f"   Output (first 200 chars): {result.stdout[:200]}")
    else:
        print(f"   ❌ magick failed with code {result.returncode}")
        print(f"   stderr: {result.stderr[:200]}")
except Exception as e:
    print(f"   ❌ Error running magick: {type(e).__name__}: {e}")

# Try to run convert --version (legacy)
print("\n4. Testing convert --version (legacy):")
try:
    result = subprocess.run(
        ["convert", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"   Return code: {result.returncode}")
    if result.returncode == 0:
        print(f"   ✅ convert works!")
        print(f"   Output (first 200 chars): {result.stdout[:200]}")
    else:
        print(f"   ❌ convert failed with code {result.returncode}")
except Exception as e:
    print(f"   ❌ Error running convert: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("SOLUTION:")
print("=" * 60)
print("""
If magick is not found:

1. Download from: https://imagemagick.org/script/download.php#windows
2. Choose: ImageMagick-7.x.x-Q16-x64-windows-dll.exe
3. Install and CHECK "Install legacy utilities"
4. Restart terminal/IDE
5. Test again with: python debug_magick.py
""")
