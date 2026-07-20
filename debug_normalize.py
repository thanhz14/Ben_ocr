#!/usr/bin/env python
# Debug normalize_formula

import traceback
from metric.cdm.modules.tokenize_latex.preprocess_formula import normalize_formula

latex_str = r"\frac{a+b}{c}"
print(f"Input: {latex_str}")

try:
    result = normalize_formula(latex_str, mode="normalize")
    print(f"Output: '{result}'")
    print(f"Output length: {len(result) if result else 0}")
    print(f"Output is empty: {not result}")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
