"""
M3 runner module entry point
Allows running as: python -m optics_sim.cli.m3_run
"""

from .m3_run import main

if __name__ == '__main__':
    import sys
    sys.exit(main())
