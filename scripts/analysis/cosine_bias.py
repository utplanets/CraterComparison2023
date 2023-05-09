import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
#sys.path.append("../scripts")
from config import *
from version2 import *
import paths
import yaml
import matplotlib.pyplot as plt

def main():
    with pd.HDFStore(dataset_name,"r") as data:
        d = plot_bias(data)
    plt.tight_layout()
    plt.savefig(paths.figures/"bias.pdf")

if __name__=="__main__":
    main()
