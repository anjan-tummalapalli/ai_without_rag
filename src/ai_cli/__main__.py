# The contents of the file: /ai_cli/ai_cli/src/ai_cli/__main__.py

import sys
from .cli import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main())