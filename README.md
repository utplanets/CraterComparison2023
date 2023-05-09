# Installation
The code in this repository is written in Python, using the anaconda distribution. The `environment.yaml` file contains a working environment to run the analysis and plot the code. Create a `conda` evironment from this file, for example by running `conda env create --file environment.yml` (or `conda env create --file environment.yml`).

# Data Download

Start by downloading the data using  `python download_data.py` from the `scripts/analysis` directory. This will download most of the data, but can't download the Benedix et al. (2020) data because of the way the journal stores the data, and doesn't currently download the `irH1k` data since the repository link isn't public yet.

# Data processing
All of these functions are from `scripts/analysis`.

  1. A later stage of the analysis uses a rescaled version of Benedix data, but the data is needed here to make sure it's part of the analaysis. Run the rescaling using `python rescale.py` which creates a new table in the `data/external` directory.
  2. The first comparison step is to normalize the catalogs so that columns are labelled in the same way.  Run `python normalize.py` to read each catalog and convert them into tables in the `interim/catalogs.hdf` file.
  3. Next, compare the catalogs against the ground truth. Run `python compare_with_iou.py` to generate one file for each catalog-comparison that includes the top 10 (by default) matches for each crater in the ground-truth, ranked by largest Intersection-over-Union (IOU) for non-zero IOU, or smallest distance for zero IOU.
  5. The previous step can be used to generate a catalog that allows duplicates. In order to remove duplicate matches (where multiple candidates match a single crater) the catalogs are iteratively pruned to remove the non-maximum matches, then regrown using remaining potential matches. The code outputs the size of the catalog as it prunes and grows. Remove the duplicates using `python deduplicate_matched.py`.

# plots
Print the table and plot the figures using the following programs. They each access the `config.py` file to get colors and (in some cases) the tables to plot.

1. `metric_table.py` : Metric table (table 1)
2. `csfd.py` : Crater Size Frequency Distribution (figure 1)
3. `errors_benedix.py` : Relative errors using the B20 method (figure 2)
4. `errors_lee.py` : Relative errors using the L19 method (figure 3)
5. `iou.py` : IOU histogram (Figure 4)
6. `tpr_lee.py` : Precision and recall for the catalogs in the L19 method (figure 5)
7. `cosine_bias.py` : Bias plot showing bias in latitude and correction (figure 6)
