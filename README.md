# scriptie2
Second Master Thesis
Wouter van Tol

# How to use
Default parameters can be adjusted by altering the dictionary in imports/params.py, or the dictionary can be altered during runtime by altering the object in code. This is also where the paths to the data, models or other folders are given.

Images can be generated using the Data class. A template is given in create_data.py.

Models are trained using alice_search.slurm which runs tune_search.py with varying arguments. 

Plots are generated using the plots.ipynb notebook.