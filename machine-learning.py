# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:40:52 2021

@author: Dpara
"""

import numpy as np;  # Numerical Python library
from matplotlib import pyplot as plt;  # Matlab-like Python module
from urllib.request import urlopen;  # importing url handling
import pandas as pd;  # importing pandas data analysis tool
import csv;

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQqwxftbS5YUDap2pGaHIYbaLEdkDuWhthRtrymKWMOS_0E7b_GmFlhFEm5bjtjVc5RnKjbrCcnH-dA/pub?gid=2024997295&single=true&output=tsv';
raw_data = urlopen(url);  # opening url
data = np.loadtxt(raw_data, skiprows=1, delimiter="\t");  # loading dataset

