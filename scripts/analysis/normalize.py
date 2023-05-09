import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))

from version2 import *
normalize_catalogs()
