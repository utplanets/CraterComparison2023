# data processing
python rescale.py -> takes the benedix catalog and rescales using dy instead of dx
python normalize.py -> takes catalogs and creates catalogs.hdf
python compare_with_iou.py -> compares the catalogs against ground truth using iou and distance
python deduplicate_matched.py -> removes duplicates from the matches and stored in the hdf file

# plots
python metric_table.py -> prints the metric table (table 1)
python csfd.py -> plots the Crater Size Frequency Distribution (figure 1)
python errors_benedix.py -> plots the relative errors using the B20 method (figure 2)
python errors_lee.py -> plots the relative errors using the L19 method (figure 3)
python iou.py -> plots the IOU histogram (Figure 4)
python tpr_lee.py -> calculates precision and recall for the catalogs in the L19 method (figure 5)
python cosine_bias.py -> plots the bias plot showing bias in latitude and correction (figure 6)


