"""Convenience script to launch the Streamlit GUI.

Server options and the telemetry opt-out live in ``.streamlit/config.toml`` so the
app behaves the same whether it is started from here or with ``streamlit run``.
"""

import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app/gui.py"], cwd=root))
