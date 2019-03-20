# Import generic libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io
from scipy import stats
import os, sys
import tqdm
import pandas as pd

# create Dataframe with the experimental info
def read_exp_info(file, MouseID):
    '''Read a csv file with information about the experiment and returns a dataFrame with data from 'MouseID'.
    file: file path with filename
    MouseID: mouse identification
    Returns
    df: a Pandas Dataframe with the info from the specified Mouse
    '''
    df = pd.read_csv(file)
    df = df.loc[df['Mouse ID'] == MouseID]

    return df
