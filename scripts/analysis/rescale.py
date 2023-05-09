import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1].absolute()))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import paths

df = pd.read_csv(paths.data/"external/ess2_502-sup-0002-2019ea001005-tdata_set_si1.csv").copy()

df["dx"] = df["Pixel value x2"] - df["Pixel value x1"]
df["dy"] = df["Pixel value y2"] - df["Pixel value y1"]

df["scale"] = df["diameter (km)"]/df.dx
df["diameter (km)"] = df["scale"]*df.dy
df.to_csv(paths.data/"external/ess_rescaled.csv")
