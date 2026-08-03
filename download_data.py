"""Download and preprocess a supported public RGB--IR dataset."""
import sys
from etfnet_cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "download-data")
    main()
