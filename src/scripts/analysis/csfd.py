import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
from config import *
from version2 import *
import paths
import yaml
import matplotlib.pyplot as plt

def main():
    catalogs = load_catalogs(Path(paths.data/"interim/catalogs.hdf"))
    naive_csfd(catalogs)
    plt.tight_layout()
    plt.savefig(paths.figures/"figure_csfd.pdf")

if __name__=="__main__":
    main()
    
