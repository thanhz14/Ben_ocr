#!/usr/bin/env python
# Debug script to see the actual error

import logging
import traceback
import sys

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add debugging to tokenize
from metric.cdm.modules.tokenize_latex.tokenize_latex import tokenize_latex

try:
    print("=" * 60)
    print("Testing tokenize_latex directly...")
    print("=" * 60)
    
    latex_str = r"\frac{a+b}{c}"
    print(f"Input LaTeX: {latex_str}")
    
    success, result = tokenize_latex(latex_str, middle_file="test_debug.txt")
    print(f"Success: {success}")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"\n❌ ERROR in tokenize_latex:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing full metric evaluation...")
print("=" * 60)

try:
    from metric.fomula import FormulaMetric
    
    metric = FormulaMetric()
    gt = r"\frac{a+b}{c}"
    pred = r"\frac{a+b}{d}"
    
    result = metric.evaluate(gt, pred, sample_id="demo_formula")
    for k, v in result.items():
        print(f"{k}: {v}")
        
except Exception as e:
    print(f"\n❌ ERROR in FormulaMetric:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
