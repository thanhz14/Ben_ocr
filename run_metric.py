#!/usr/bin/env python
import os

# Add ImageMagick to PATH
os.environ['PATH'] = r'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI' + os.pathsep + os.environ['PATH']

# Verify
import subprocess
try:
    result = subprocess.run(["magick", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ ImageMagick loaded successfully!")
    else:
        print("❌ ImageMagick NOT found!")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Run metric
from metric.fomula import FormulaMetric

print("\nRunning metric evaluation...\n")
metric = FormulaMetric()
gt = r"\frac{a+b}{c}"
pred = r"\frac{a+b}{d}"
result = metric.evaluate(gt, pred, sample_id="demo_formula")

print("\n" + "=" * 50)
print("RESULTS:")
print("=" * 50)
for k, v in result.items():
    print(f"{k}: {v}")
