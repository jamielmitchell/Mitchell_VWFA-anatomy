# Mitchell_DCN2026
Code associated with:
Mitchell, J. L., Yablonski, M., Jimenez, M., Chiu, H., & Yeatman, J. D. (2026). Anatomy of the Visual Word Form Area in Dyslexia. bioRxiv. [https://doi.org/10.64898/2026.07.31.742142](https://doi.org/10.64898/2026.07.31.742142)

## General Note
The data associated with this manuscript and analysis can be found in a Stanford University Libraries Digital Repository at [https://purl.stanford.edu/qq284rg5214](https://purl.stanford.edu/qq284rg5214). In order for the code to run flawlessly, you should download this data and store it in a folder named `data` and then download the code and store it in a parent folder alongside `data`.

## Content and Instructions
- ROI_size.ipynb: A jupyter notebook that extracts and analyzes size metrics from ROIs.
- define_centers.ipynb: A jupyter notebook that calculates the center point for ROIs.
- coordinate_locations.ipynb: A jupyter notebook that extracts and analyzes coordinate location infromation from ROI center points. Requires define_centers.ipynb output.
- center distances.ipynb: A jupyter notebook that calculates and analyzes average spread of ROIs. Requires coordinate_locations.ipynb output.
- wm_density.ipynb: A jupyter notebook that extracts average white matter bundle densities from ROIs.
- utils.py: A python script that contains custom functions to help make things a little easier.

*ROIs == Regions of Interest