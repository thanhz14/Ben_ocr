#!/usr/bin/env python
# Debug normalize_formula step by step

import traceback
from metric.cdm.modules.tokenize_latex.preprocess_formula import preprocess_line, likely_bad_latex
from metric.cdm.modules.tokenize_latex.parse_guard import suppress_pylatexenc_warnings
from metric.cdm.modules.tokenize_latex.pylatexenc_to_katex import parse_latex_to_katex_ast
from metric.cdm.modules.tokenize_latex.katex_renderer import KaTeXRenderer
from metric.cdm.modules.tokenize_latex.options import Options

latex_str = r"\frac{a+b}{c}"
print(f"Input: {latex_str}\n")

try:
    # Step 1: preprocess_line
    print("Step 1: preprocess_line")
    guard_input = preprocess_line(latex_str, keep_dollar=True)
    print(f"  Output: '{guard_input}'")
    
    # Step 2: likely_bad_latex check
    print("\nStep 2: likely_bad_latex check")
    is_bad, reason = likely_bad_latex(guard_input, latex_type="formula")
    print(f"  is_bad: {is_bad}")
    print(f"  reason: {reason}")
    
    if is_bad:
        print("  ❌ LaTeX detected as BAD - returning empty!")
    else:
        # Step 3: Replace $
        print("\nStep 3: Replace $")
        pre = guard_input.replace("$", " ")
        print(f"  Output: '{pre}'")
        
        # Step 4: normalize_rm
        print("\nStep 4: normalize_rm")
        from metric.cdm.modules.tokenize_latex.preprocess_formula import normalize_rm
        pre = normalize_rm(pre)
        print(f"  Output: '{pre}'")
        
        # Step 5: parse_latex_to_katex_ast
        print("\nStep 5: parse_latex_to_katex_ast")
        with suppress_pylatexenc_warnings():
            ast = parse_latex_to_katex_ast(pre)
            print(f"  AST type: {type(ast)}")
            print(f"  AST: {ast}")
            
            # Step 6: KaTeXRenderer
            print("\nStep 6: KaTeXRenderer.render")
            renderer = KaTeXRenderer(array_mode="formula")
            out = renderer.render(ast, Options())
            print(f"  Output: '{out}'")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
