%load_ext autoreload
%autoreload 2
%matplotlib inline

# Import generic libraries
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd
from my_utils import *
from plotting import *
from pynalysis import utils
import pickle

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/trackedROIs/'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)
n_unit = rois.shape[0]
fig_list = {}
for i in range(n_unit):
    f = plot_tracked_unit(i, exp_data, info_df, rois)
    fig_list.update({'Unit '+str(i): f})

save_figs(fig_list, mouse_id, 'all', 'all', save_path)

fig_list = plot_trialframes(exp_data.trialArray, exp_data.stimulus)
save_figs(fig_list, mouse_id, date, str(session_info.Session), save_path)

fig_list1 = plotData(data[:,sess_rois], run_tracked, stim_start)
print(fig_list1.items)
save_figs(fig_list1, mouse_id, date, str(session_info.Session), save_path+'trackedROIs/running/')

fig_list2 = plot_trialframes(trials[:,:,sess_rois], stim_start)
save_figs(fig_list2, mouse_id, date, str(session_info.Session), save_path+'trackedROIs/')
