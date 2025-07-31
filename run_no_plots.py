#!/usr/bin/env python3
"""
Quick script to run diagnostics without plots.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.run_diagnostics import run_analysis_with_plot_control

if __name__ == "__main__":
    print("Running diagnostics without plots...")
    results = run_analysis_with_plot_control(show_plots=False)
    print("Analysis completed successfully!")
    print(f"Analyzed {len(results.get('analyzed_isbns', []))} books") 