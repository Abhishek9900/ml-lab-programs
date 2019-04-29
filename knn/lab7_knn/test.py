# Importing libraries
import pandas as pd
import numpy as np
import math
import operator

#### Start of STEP 1

# Importing data
data = pd.read_csv("knn-wisc_bc_data.csv")

print(data.info())
print(data.describe())