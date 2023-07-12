import json, zipfile
import pandas as pd
import urllib.request
import yaml
import paths
from io import BytesIO

def process_robbins_data(source):
    filename="RobbinsCraterDatabase_20120821/RobbinsCraters_tab.txt.gz"
    with zipfile.ZipFile(source) as archive:
        data = BytesIO(archive.read(filename))
        Robbins = pd.read_csv(data, sep="\t",encoding='latin-1',low_memory=False, compression='gzip')
    return Robbins

def process_lagain_data(source):
    zipfilename="Crater_counting/lagain_db_cratertools_filtered.json.zip"
    filename="lagain_db_cratertools_filtered.json"
    with zipfile.ZipFile(source) as archive:
        zipdata = BytesIO(archive.read(zipfilename))
        with zipfile.ZipFile(zipdata) as djszip:
            djs = djszip.read(filename)
            dj = json.loads(djs)
            rows=[]
            for _entry in dj["features"]:
                entry = _entry["properties"]
                rows.append([entry["x_coord"],entry["y_coord"],entry["Diam_km"]])
    Lagain = pd.DataFrame(rows, columns=["Long","Lat","Diameter (km)"])
    return Lagain

def process_salamuniccar_data(source):
    filename="GoranSalamuniccar_MarsCraters/MA132843GT/original_files/MA132843GT.xlsx"
    with zipfile.ZipFile(source) as archive:
        data = BytesIO(archive.read(filename))
        Sal = pd.read_excel(data)
    return Sal

def read_catalog(filename, input_dims, raw=False,function=None,**kwargs):
    """Attempts to read the catalog from filename with dimensions input_dims
    returns a normalized catalog with correct column names
    """
    print(f"Reading {filename}")
    function_dict = dict(Robbins=process_robbins_data,
                         Lagain=process_lagain_data,
                         Salamuniccar=process_salamuniccar_data)
    print(filename, function)
    if function is not None:
        data = function_dict[function](filename)
    elif filename.suffix == ".tsv":
        data = pd.read_csv(filename, sep="\t",encoding='latin-1',**kwargs)
    else:
        data = pd.read_csv(filename, **kwargs)
    if raw:
        return data
    print(filename, data.columns)
    data = data[input_dims].copy()
    output_dims=["Long","Lat","Diameter (km)"]
    data = data.rename(columns=dict(zip(input_dims, output_dims)))
    return data


def sha256(filename,theirs,BUF_SIZE=65536):
    
    import hashlib
    digest = hashlib.sha256()
    with open(filename,"rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            digest.update(data)
    mine = digest.hexdigest()  

    if mine != theirs:
        print(f"SHA256 mismatch for {filename}: {mine} {theirs}")
        return False
    return True

def download():

    catalogs = yaml.safe_load(open(paths.data/"catalogs.yaml",'r'))
    for name,entry in catalogs.items():
        pf = (paths.data/"external")/entry["filename"]
        
        if not pf.exists() and entry["download"]:
            #download
            print("Downloading {} to {}".format(entry["url"],pf))
            urllib.request.urlretrieve(entry["url"],str(pf))
        elif not pf.exists() and not entry["download"]:
            if "message" in entry:
                print(entry["message"])
            else:
                print("Cannot download {}".format(entry["filename"]))
            print("\tURL : ", entry["url"])
        else:
            print("File {} already exists".format(pf))
        if pf.exists():
            sha256(pf,entry.get("sha256",None))
        else:
            print("File {} missing".format(pf))

def normalize_catalogs():
    catalogs = yaml.safe_load(open(paths.data/"catalogs.yaml",'r'))
    dest = (paths.data/"interim")
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    for name,row in catalogs.items():
        data = read_catalog((paths.data/"external")/row["filename"], row["xyz"],function=row.get("function",None))
        print("Saving data to interim/catalogs.hdf")
        data.to_hdf(paths.data/"interim/catalogs.hdf",key=name,append=False)
