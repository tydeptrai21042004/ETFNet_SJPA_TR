"""Resume ETFNet-SJPA-TR. Equivalent to the ``resume`` CLI subcommand."""
import sys
from etfnet_cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'resume')
    main()
