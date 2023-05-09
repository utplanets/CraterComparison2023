import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
#sys.path.append("../scripts")
from config import *
from version2 import *
import paths
import yaml
import matplotlib.pyplot as plt


if __name__=="__main__":
    import sys
    #filename
    catalogs = load_catalogs(paths.data/"interim/catalogs.hdf")
    data = pd.HDFStore(dataset_name,'r')
    if len(sys.argv)>1:
        print_metric_table(data,catalogs, form=sys.argv[1])
    else:
        print_metric_table(data,catalogs, form="latex")
    data.close()
