%load_ext autoreload
%autoreload 2
%matplotlib inline

# preprocess.py
# Preprocessing of imaging data structure from hfapc task
# Splits the data into trials
# Trials are selected acording to first lick after delivery

# Import generic libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io
from scipy import stats
import os, sys
import tqdm
import pandas as pd
from my_utils import *
from pynalysis import utils
import pickle
from plotting import *
# Import local tools
sys.path.append(os.path.expanduser('/Users/anabottura/github/pynalysis/'))

# Setup path variables - should move this into another file later
mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
infofile = output+'exp_info.csv'
filepath = '/Users/anabottura/Documents/2p_analysis/data/pipeline_output/imaging/'
rois_file = '/Users/anabottura/Documents/2p_analysis/data/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
plot = 'True'
################################################################################
# Read the information about the experiment
info_df = read_exp_info(infofile, mouse_id)
info_df = info_df[2:] #remove the pretraining Sessions --- need to change this

# read csv file with tracked ROIs
names = []
for s, name in enumerate(info_df.Date.values):
    names.append(name+'_'+str(info_df.Session.values[s]))
rois = pd.read_csv(rois_file, header=None, names=names)

exp_data = pd.DataFrame(names, columns=['Sessions'])
comp = info_df.loc[:,['Compound Code', 'Compound ID']]
comp.index = range(len(comp))
exp_data = pd.concat([comp, exp_data], axis=1)
exp_data = exp_data.set_index('Sessions')

# run the preprocessing and split in trials, find running trials and add everything to DataFrame
df = pd.DataFrame()
for i in range(info_df.shape[0]):
    session_info = info_df.iloc[i]
    date = session_info.Date
    session = "area"+str(session_info.Session)
    dateSess = date+'_'+str(session_info.Session)
    sess_rois = rois[dateSess]

    data, trials, trials_windows, stim_start, licks= preprocess(mouse_id, date, session, filepath)
    run_trials, trial_speed = find_run(trials_windows, mouse_id, date, session, filepath)
    running_trials = trials[run_trials==1,:,:]
    run_tracked = running_trials[:,:,sess_rois]

    s = pd.Series({'Data': data,'trialArray': trials,
    'trial_limit': trials_windows, 'stimulus': stim_start, 'running_t': run_trials,
    'trial_speed': trial_speed, 'Licks': licks}, name=dateSess)
    df = df.append(s)

    if plot == 'True':
        fig_list = plot_trialframes(trials, stim_start)
        save_figs(fig_list, mouse_id, date, str(session_info.Session), save_path)

        fig_list1 = plotData(data[:,sess_rois], run_tracked, stim_start)
        print(fig_list1.items)
        save_figs(fig_list1, mouse_id, date, str(session_info.Session), save_path+'trackedROIs/running/')

        fig_list2 = plot_trialframes(trials[:,:,sess_rois], stim_start)
        save_figs(fig_list2, mouse_id, date, str(session_info.Session), save_path+'trackedROIs/')


df.index.name = 'Sessions'
exp_data = pd.concat([exp_data, df], axis=1)

#save exp_data as a picked file to output filepath

filename = output+'%s_exp_data.pickle'%mouse_id
with open(filename, 'wb') as f:
    pickle.dump(exp_data, f)
