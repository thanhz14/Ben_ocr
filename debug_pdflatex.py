#!/usr/bin/env python
# Debug pdflatex detection

from metric.cdm.modules.texlive_env import (
    resolve_tex_binary,
    describe_tex_runtime,
    build_tex_env,
)
import shutil
import os

print("=" * 60)
print("Checking TeX/LaTeX Setup")
print("=" * 60)

# Check if pdflatex is in PATH
print("\n1. Checking if pdflatex is in system PATH:")
which_pdflatex = shutil.which("pdflatex")
print(f"   shutil.which('pdflatex'): {which_pdflatex}")

# Check resolved pdflatex
print("\n2. Checking resolved pdflatex from CDM:")
resolved_pdflatex = resolve_tex_binary("pdflatex")
print(f"   resolve_tex_binary('pdflatex'): {resolved_pdflatex}")

# Check resolved kpsewhich
print("\n3. Checking resolved kpsewhich from CDM:")
resolved_kpsewhich = resolve_tex_binary("kpsewhich")
print(f"   resolve_tex_binary('kpsewhich'): {resolved_kpsewhich}")

# Check tex environment
print("\n4. Full TeX runtime description:")
tex_runtime = describe_tex_runtime()
for key, value in tex_runtime.items():
    print(f"   {key}: {value}")

# Check build_tex_env
print("\n5. Build TeX environment:")
tex_env = build_tex_env()
print(f"   PATH (first 100 chars): {tex_env.get('PATH', '')[:100]}...")
print(f"   CDM_PDFLATEX: {tex_env.get('CDM_PDFLATEX', 'NOT SET')}")
print(f"   CDM_KPSEWHICH: {tex_env.get('CDM_KPSEWHICH', 'NOT SET')}")
print(f"   TEXMFCNF: {tex_env.get('TEXMFCNF', 'NOT SET')}")

# Try to run pdflatex --version
print("\n6. Testing pdflatex --version:")
import subprocess
try:
    result = subprocess.run(
        ["pdflatex", "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        env=tex_env
    )
    print(f"   Return code: {result.returncode}")
    if result.returncode == 0:
        print(f"   ✅ pdflatex works!")
        print(f"   Output (first 200 chars): {result.stdout[:200]}")
    else:
        print(f"   ❌ pdflatex failed with code {result.returncode}")
        print(f"   stderr: {result.stderr[:200]}")
except Exception as e:
    print(f"   ❌ Error running pdflatex: {type(e).__name__}: {e}")
