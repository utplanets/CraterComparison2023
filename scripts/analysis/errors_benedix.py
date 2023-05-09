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
    ranges = {"Long":np.linspace(-2,2,501),
              "Lat":np.linspace(-2,2,501),
              "Diameter":np.linspace(-50,50,501),
          }
    
    keys=[f"/Benedix/{k}_Robbins/without_duplicates" for k in plot_names]
    _xlim=(-180,180)
    _ylim=(-65,65)
    _dlim=(1.5,10)
    with pd.HDFStore(dataset_name,"r") as data:
        d = plot_distance(data,keys,ranges=ranges,xlim=_xlim,ylim=_ylim,dlim=_dlim)
        plt.tight_layout()
        plt.savefig(paths.figures/"errors_benedix.pdf")

if __name__=="__main__":
    main()
