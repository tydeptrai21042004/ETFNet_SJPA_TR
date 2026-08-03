"""Run paired RGB--IR inference. Equivalent to the ``predict`` CLI subcommand."""
import sys
from etfnet_cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'predict')
    main()
