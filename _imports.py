import os
import sys
import re
import math
import random
import json
import warnings
from pprint import pprint
from collections import *
from typing import Any

import plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd

import numpy as np
import numpy.typing as npt

from pandas.plotting import *

import scipy
import scipy.stats
import scipy.ndimage
import scipy.sparse

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import seaborn.objects as so

import joblib

import sklearn
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.base import *
from sklearn.metrics import *
from sklearn.metrics.pairwise import *
from sklearn.impute import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.compose import *
from sklearn.utils.validation import *
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.tree import *
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.dummy import *
from sklearn.calibration import *
from sklearn.svm import *
from sklearn.multiclass import *
from sklearn.neighbors import *
from sklearn.multioutput import *
from sklearn.decomposition import *

np.random.seed = 110
random.seed = 110

sklearn.set_config(transform_output="pandas")

sns.set_theme(style="ticks", rc={"figure.figsize": (5, 5)})
so.Plot.config.theme.update(sns.axes_style(style="ticks", rc={"figure.figsize": (5, 5)}))

pio.templates.default = "seaborn"
# pio.templates.default = "plotly_dark"
# plt.style.use("dark_background")

pd.options.plotting.backend = "matplotlib"

plt.rcParams["figure.figsize"] = (5, 5)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)
