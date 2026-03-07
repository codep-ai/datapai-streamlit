"""pytest configuration — add project root to sys.path."""
import sys
import os

# Ensure the project root is on the path so 'traceability' package is found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
