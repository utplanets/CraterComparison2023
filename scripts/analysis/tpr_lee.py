import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
from config import *
from version2 import *
import paths
import yaml
import matplotlib.pyplot as plt


def main():
    keys=[f"/Lee/{k}_Robbins/without_duplicates" for k in plot_names]
    catalogs = load_catalogs(paths.data/"interim/catalogs.hdf")

    with pd.HDFStore(dataset_name,"r") as data:
        axs = plot_metric_vs_Lat(data,catalogs,keys,dlim=(1.5,10),ranges=None)
        axs = plot_metric_vs_Lat(data,catalogs,keys,dlim=(1.5,10),precision=True, ls="--",axs=axs,ranges=None,alpha=0.5)
#        axs = plot_metric_vs_Lat(data,catalogs["Robbins"],keys,dlim=(4,10),ls="--",axs=axs,ranges=None,alpha=0.25)
        for a in axs:
            a.grid()
    plt.tight_layout()
    plt.savefig(paths.figures/"tpr_lee.pdf")


if __name__=="__main__":
    main()
