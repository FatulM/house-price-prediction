# Imports:

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
from pathlib import Path

import numpy as np
import numpy.typing as npt
import numpy.lib
import numpy.random
import numpy.linalg
import numpy.fft
import numpy.polynomial

import pandas as pd
from pandas.plotting import *

import scipy
import scipy.cluster
import scipy.constants
import scipy.datasets
import scipy.fft
import scipy.fftpack
import scipy.integrate
import scipy.interpolate
import scipy.io
import scipy.linalg
import scipy.misc
import scipy.ndimage
import scipy.odr
import scipy.optimize
import scipy.signal
import scipy.sparse
import scipy.spatial
import scipy.special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import seaborn.objects as so

import plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff

import sklearn
from sklearn.base import *
from sklearn.calibration import *
from sklearn.cluster import *
from sklearn.compose import *
from sklearn.covariance import *
from sklearn.cross_decomposition import *
from sklearn.datasets import *
from sklearn.datasets.data import *
from sklearn.datasets.images import *
from sklearn.datasets.descr import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.exceptions import *
from sklearn.experimental import *
from sklearn.externals import *
from sklearn.feature_extraction import *
from sklearn.feature_extraction.image import *
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.gaussian_process.kernels import *
from sklearn.impute import *
from sklearn.inspection import *
from sklearn.isotonic import *
from sklearn.kernel_approximation import *
from sklearn.kernel_ridge import *
from sklearn.linear_model import *
from sklearn.manifold import *
from sklearn.metrics import *
from sklearn.metrics.cluster import *
from sklearn.metrics.pairwise import *
from sklearn.mixture import *
from sklearn.model_selection import *
from sklearn.multiclass import *
from sklearn.multioutput import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.random_projection import *
from sklearn.semi_supervised import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.utils import *
from sklearn.utils.arrayfuncs import *
from sklearn.utils.class_weight import *
from sklearn.utils.deprecation import *
from sklearn.utils.discovery import *
from sklearn.utils.estimator_checks import *
from sklearn.utils.extmath import *
from sklearn.utils.fixes import *
from sklearn.utils.graph import *
from sklearn.utils.metadata_routing import *
from sklearn.utils.metaestimators import *
from sklearn.utils.multiclass import *
from sklearn.utils.murmurhash import *
from sklearn.utils.optimize import *
from sklearn.utils.sparsefuncs import *
from sklearn.utils.sparsefuncs_fast import *
from sklearn.utils.stats import *
from sklearn.utils.validation import *

# Configs:

sns.set_theme(style="ticks", rc={"figure.figsize": (5, 5)})
so.Plot.config.theme.update(sns.axes_style(style="ticks", rc={"figure.figsize": (5, 5)}))

# pio.templates.default = "plotly_dark"
# plt.style.use("dark_background")

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

Path("input/").mkdir(parents=True, exist_ok=True)
Path("input/docs/").mkdir(parents=True, exist_ok=True)
Path("data/").mkdir(parents=True, exist_ok=True)
Path("output/").mkdir(parents=True, exist_ok=True)
