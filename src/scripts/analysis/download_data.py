import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
from config import *
from version2 import *
import paths
import yaml
import matplotlib.pyplot as plt

def main():
    if not (paths.data/"external").exists():
        (paths.data/"external").mkdir(parents=True,exist_ok=True)
    download()

if __name__=="__main__":
    main()
