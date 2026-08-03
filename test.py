"""Validate ETFNet-SJPA-TR. Equivalent to the ``val`` CLI subcommand."""
import sys
from etfnet_cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'val')
    main()
