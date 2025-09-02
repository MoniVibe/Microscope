"""
Make M3 runner accessible as: python -m optics_sim.cli.m3_run
"""

import sys
from pathlib import Path

# Add src directory to path for proper imports
src_dir = Path(__file__).parent.parent.parent.parent / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from optics_sim.cli.m3_run import main

if __name__ == '__main__':
    sys.exit(main())
