"""
Development entry point for the ORBIT Platform.
"""

import sys
import os

# Ensure local repository takes precedence
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  O.R.B.I.T. // BI-TEMPORAL RECONNAISSANCE TERMINAL")
    print("  Development Server Active: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
