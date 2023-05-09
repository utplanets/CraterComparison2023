import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))

from interface import *
from config import *
from version2 import *
from common_code import *
#from analysis_code import *
import paths

from tqdm import tqdm

from joblib import Parallel, delayed
import pickle

def process(catalog_filename, catalogs_to_compare):
    catalogs = load_catalogs(catalog_filename)


    method = "Benedix"
    for catalog_name,truth_name in catalogs_to_compare:
        print(catalog_name, truth_name)
        filename=paths.data/f"interim/{method}_{catalog_name}_{truth_name}.pkl"
        if filename.exists():
            print("Filename exists, won't overwrite")
            continue
        cr = catalogs[truth_name]
        data = Parallel(n_jobs=32)(delayed(iouD_topN)(row.Long,row.Lat,row["Diameter (km)"],
                                                      cr.Long.values,cr.Lat.values, cr["Diameter (km)"].values,2.,2.,50.) for irow, row in tqdm(catalogs[catalog_name].iterrows()))
        pickle.dump(data,open(filename,'wb'))

    method = "Lee"
    for catalog_name,truth_name in catalogs_to_compare:
        print(catalog_name, truth_name)
        filename=paths.data/f"interim/{method}_{catalog_name}_{truth_name}.pkl"
        if filename.exists():
            print("Filename exists, won't overwrite")
            continue
        cr = catalogs[truth_name]
        data = Parallel(n_jobs=32)(delayed(iouD_topN)(row.Long,row.Lat,row["Diameter (km)"],
                                                      cr.Long.values,cr.Lat.values, cr["Diameter (km)"].values,25.,25.,25.,method="Lee") for irow, row in tqdm(catalogs[catalog_name].iterrows()))
        pickle.dump(data,open(filename, 'wb'))


if __name__=="__main__":
    process(paths.data/"interim/catalogs.hdf", catalogs_to_compare)
