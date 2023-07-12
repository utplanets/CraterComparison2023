import paths
class config(object):
    R_planet=3391.0

cfg = config()

catalogs = ['Benedix','DeepMars1Mars', 'DeepMars2',
            "BeRescaled","irH1k"]
human = ["Robbins","Salamuniccar"]
cols = ['Long', 'Lat', 'Diameter (km)']


plot_names = ["Benedix","DeepMars2","irH1k", "BeRescaled"]
labels = dict(BeRescaled="B_Rescaled", Long="Longitude",Lat="Latitude",
              DeepMars1Mars="DeepMars1")

#import palettable
#allcolors = palettable.cartocolors.qualitative.get_map("Bold_10").hex_colors
colors =      {"Benedix": "#7F3C8D",
              "DeepMars1Mars": "#11A579",
              "DeepMars1Moon": "#3969AC",
              "DeepMars2": "#F2B701",
              "BeRescaled": "#E73F74",
              "irH1k": "#80BA5A",
              "Lagain": "#008695",
              "Robbins": "#CF1C90",
              "Salamuniccar": "#F97B72"}

names = catalogs + human
#colors = dict(zip(names, allcolors))

#---
catalog_file = paths.data/"interim/catalogs.hdf"
methods = ["Benedix","Lee"]
catalogs_to_compare = [("DeepMars2","Robbins"),
 ("Benedix","Robbins"),
 ("BeRescaled","Robbins"),
# ("Lagain","Robbins"),
 ("irH1k","Robbins"),
        ]

dataset_name = paths.data/"processed/v2_comparison_data_-180_180_-90_90_0_1000.h5"
