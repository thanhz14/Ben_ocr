#!/usr/bin/env python
import os
import subprocess
import shutil

print("=" * 70)
print("DEBUG: Finding which command fails")
print("=" * 70)

# Add ImageMagick to PATH first
os.environ['PATH'] = r'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI' + os.pathsep + os.environ['PATH']

# Test each command
print("\n1. Testing magick:")
try:
    result = subprocess.run(["magick", "--version"], capture_output=True, text=True, timeout=5)
    print(f"   ✅ magick works (code: {result.returncode})")
except Exception as e:
    print(f"   ❌ magick failed: {e}")

print("\n2. Testing pdflatex:")
try:
    result = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True, timeout=5)
    print(f"   ✅ pdflatex works (code: {result.returncode})")
except Exception as e:
    print(f"   ❌ pdflatex failed: {e}")

print("\n3. Testing pdftoppm:")
try:
    result = subprocess.run(["pdftoppm", "--version"], capture_output=True, text=True, timeout=5)
    print(f"   ✅ pdftoppm works (code: {result.returncode})")
except Exception as e:
    print(f"   ❌ pdftoppm failed: {e}")

print("\n4. Testing identify:")
try:
    result = subprocess.run(["identify", "--version"], capture_output=True, text=True, timeout=5)
    print(f"   ✅ identify works (code: {result.returncode})")
except Exception as e:
    print(f"   ❌ identify failed: {e}")

# Check PATH
print("\n5. Current PATH:")
paths = os.environ['PATH'].split(os.pathsep)
for p in paths[:5]:
    print(f"   {p}")
print("   ...")

print("\n" + "=" * 70)
print("Which command fails?")
print("=" * 70)
