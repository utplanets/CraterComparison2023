import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))

from interface import *
from config import *
from version2 import *
from common_code import *
import paths

from tqdm import tqdm

from joblib import Parallel, delayed
import pickle



def process():
    catalog_file = paths.data/"interim/catalogs.hdf"
    match_catalogs(catalog_file, methods, catalogs_to_compare)

if __name__=="__main__":
    process()
