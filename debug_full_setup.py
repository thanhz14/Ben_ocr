#!/usr/bin/env python
# Full debug to find ImageMagick and setup PATH correctly

import os
import sys
import subprocess
import glob

print("=" * 70)
print("FULL IMAGEMAGICK & PATH DEBUG")
print("=" * 70)

# Check common ImageMagick installation paths
print("\n1. Scanning common ImageMagick installation paths:")
common_paths = [
    r"C:\Program Files\ImageMagick*",
    r"C:\Program Files (x86)\ImageMagick*",
    r"C:\ImageMagick*",
]

found_paths = []
for pattern in common_paths:
    matches = glob.glob(pattern)
    if matches:
        for match in matches:
            print(f"   ✅ Found: {match}")
            found_paths.append(match)
            
            # Check if magick.exe exists
            magick_exe = os.path.join(match, "magick.exe")
            if os.path.exists(magick_exe):
                print(f"      └─ magick.exe: YES ✅")
            else:
                print(f"      └─ magick.exe: NO ❌")

if not found_paths:
    print("   ❌ ImageMagick NOT FOUND in common paths!")
    print("   → You need to install ImageMagick first!")
    print("   → Download from: https://imagemagick.org/script/download.php#windows")
    sys.exit(1)

print(f"\n2. Current PATH in Python:")
print(f"   {os.environ.get('PATH', 'NOT SET')[:150]}...")

print(f"\n3. Try to add ImageMagick to PATH and test:")
for img_path in found_paths:
    os.environ['PATH'] = img_path + os.pathsep + os.environ.get('PATH', '')
    
    print(f"\n   Testing with: {img_path}")
    try:
        result = subprocess.run(
            ["magick", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ SUCCESS! magick works!")
            print(f"      Output: {result.stdout.split(chr(10))[0]}")
            
            print(f"\n" + "=" * 70)
            print(f"SOLUTION:")
            print(f"=" * 70)
            print(f"""
Add this line to your Python script BEFORE importing metric:

    import os
    os.environ['PATH'] = r'{img_path}' + os.pathsep + os.environ['PATH']

Or run this in PowerShell before python:

    $env:Path = '{img_path};' + $env:Path
    python -m metric.fomula
""")
            break
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}")

print("=" * 70)
