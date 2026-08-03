"""Export ETFNet-SJPA-TR. Equivalent to the ``export`` CLI subcommand."""
import sys
from etfnet_cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'export')
    main()
