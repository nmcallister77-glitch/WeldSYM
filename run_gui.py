"""Convenience script to launch the Streamlit GUI."""

import os
import sys

# Allow running from repo root
os.chdir(os.path.dirname(__file__))

sys.exit(
    os.system(
        "streamlit run app/gui.py --server.headless true --server.address localhost --server.port 8501"
    )
)
