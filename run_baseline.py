#!/usr/bin/env python
"""
Quick run script for baseline inference
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.baseline_inference import main

if __name__ == "__main__":
    # Set default arguments if none provided
    if len(sys.argv) == 1:
        sys.argv.extend(['--task', 'medium', '--episodes', '1'])
    
    main()
