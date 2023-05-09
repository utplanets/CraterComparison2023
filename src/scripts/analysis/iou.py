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
    keys=[f"{k}_Robbins" for k in plot_names]
    with pd.HDFStore(dataset_name,"r") as data:
        d = plot_iou_density(data,keys,iou_range=None)
    plt.tight_layout()
    plt.savefig(paths.figures/"iou.pdf")

if __name__=="__main__":
    main()
