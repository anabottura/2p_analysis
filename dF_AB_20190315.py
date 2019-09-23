%load_ext autoreload
%autoreload 2
%matplotlib inline

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

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/Correlations/'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)

# # Import local tools
# sys.path.append(os.path.expanduser('/Users/anabottura/github/pynalysis/'))
#
#
# infofile = '/Users/anabottura/Documents/2p_analysis/data/exp_info.csv'
# mouse_id = "CTBD7.1d"
# filepath = '/Users/anabottura/Documents/2p_analysis/data/pipeline_output/imaging/'
# rois_file = '/Users/anabottura/Documents/2p_analysis/data/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
# save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/trackedROIs/running/'%mouse_id
# output = '/Users/anabottura/Documents/2p_analysis/data/'
# info_df = read_exp_info(infofile, mouse_id)
# info_df = info_df[2:]
#
# # read csv file with tracked ROIs
# names = []
# for s, name in enumerate(info_df.Date.values):
#     names.append(name+'_'+str(info_df.Session.values[s]))
# rois = pd.read_csv(rois_file, header=None, names=names)
#
# exp_data = pd.DataFrame(names, columns=['Sessions'])
# comp = info_df.loc[:,['Compound Code', 'Compound ID']]
# comp.index = range(len(comp))
# exp_data = pd.concat([comp, exp_data], axis=1)
# exp_data = exp_data.set_index('Sessions')
#
# a, b, c, d = preprocess('CTBD7.1d', '2019/02/01', 'area2', filepath)
# # run the preprocessing and split in trials, find running trials and add everything to DataFrame
# df = pd.DataFrame()
# plot = 'False'
# for i in range(info_df.shape[0]):
#     session_info = info_df.iloc[i]
#     date = session_info.Date
#     session = "area"+str(session_info.Session)
#     dateSess = date+'_'+str(session_info.Session)
#     sess_rois = rois[dateSess]
#     sess_rois
#
#     data, trials, trials_windows, stim_start= preprocess(mouse_id, date, session, filepath)
#     run_trials, trial_speed = find_run(trials_windows, mouse_id, date, session, filepath)
#     a = trials[run_trials==1,:,:]
#     b = a[:,:,sess_rois]
#
#     s = pd.Series({'Data': data,'trialArray': trials,
#     'trial_limit': trials_windows, 'stimulus': stim_start, 'running_t': run_trials,
#     'trial_speed': trial_speed}, name=dateSess)
#     df = df.append(s)
#
#     if plot == 'True':
#         fig_list = plotData(data[:,sess_rois], b, stim_start)
#         save_figs(fig_list, mouse_id, date, str(session_info.Session), save_path)
#
# df.index.name = 'Sessions'
# exp_data = pd.concat([exp_data, df], axis=1)
#
# #save exp_data as a picked file to output filepath
# filename = output+'%s_exp_data.pickle'%mouse_id
# with open(filename, 'wb') as f:
#     pickle.dump(exp_data, f)

################################################################################
# # to get random intervals outside of the trial interval
# sample_size = 60 #size of the window I will take in frames
# windows = exp_data.trial_limit
# stimulus = exp_data.stimulus
# sample_size_df = stimulus.copy()
# sample_size_df[:] = sample_size #create a df with the value of the window
# #reshape the trial limits to have it in a dataframe
# a=pd.DataFrame([np.array([r[0]]) for r in windows], index=windows.index, columns=['trial_start'])
# b=pd.DataFrame([np.array([r[1]]) for r in windows], index=windows.index, columns=['trial_end'])
# windows = pd.concat([a,b],axis=1)
#
# to_sub = sample_size_df*[np.ones(r[0].shape) for r in windows['trial_start']]
# w_end = windows.trial_start.subtract(to_sub)
# w_end = w_end.subtract(to_sub)
# w_start = windows.trial_start.add(to_sub)
# b_window = pd.concat([w_start,w_end],axis=1)
#
# random_ints = pd.DataFrame(index=b_window.index, columns=['start_random', 'start_pre'])
# for date in b_window.index:
#     a = b_window.loc[date]
#     b = windows.trial_start[date]
#     start_rn = a.iloc[0][0][:-1]
#     end_rn = a.iloc[1][0][1:]
#     n_start_rn = start_rn[end_rn-start_rn>0]
#     n_end_rn = end_rn[end_rn-start_rn>0]
#     start_pre = b[0][1:]
#     n_start_pre = start_pre[end_rn-start_rn>0]
#     br = [np.random.randint(s, e) for s,e in zip(n_start_rn,n_end_rn)]
#     random_ints.loc[date]['start_random'] = pd.Series(br, name=date)
#     random_ints.loc[date]['start_pre'] = pd.Series(n_start_pre, name=date)

################################################################################

# sample_size = int(exp_data.stimulus[0])
# windows = exp_data.trial_limit
# stimulus = exp_data.stimulus
# start1=pd.DataFrame([pd.Series(r[0]) for r in windows])
# start1.index = windows.index
# start2 = pd.DataFrame([pd.Series(r[0]).add(sample_size*np.ones(r[0].shape)) for r in windows])
# start2.index = windows.index
#
# f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.trapz)
# np.median
# np.max
# spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
# name = 'pre_post_area'
# if not os.path.exists(spath):
#     os.makedirs(spath)
# f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
# f.clf()
#
# sample_size = 60
# start1 = random_ints.start_random
# start2 = random_ints.start_pre
# f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.mean)
# f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.median)
# f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.max)
# f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.trapz)
#
# spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
# name = 'rdm_pre_max'
# if not os.path.exists(spath):
#     os.makedirs(spath)
# f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
# f.clf()
########################################################################################



def plot_tracked_unit(unit):
    gb = info_df.groupby('Date')
    order = ['B', 'Y', 'A', 'X']
    grouped = exp_data.groupby('Compound Code')

    f = plt.figure(figsize=(28,20))
    f.suptitle('Tracked Unit '+str(unit+1))
    f.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Time")
    plt.ylabel("dF/F")
    for i, date in enumerate(gb.groups.keys()):
        group = gb.get_group(date)
        group = group.set_index('Compound Code')
        ia = f.add_subplot(1,gb.ngroups,i+1, frameon=False)
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ia.set_title((str(date)+'\n'+'\n'), fontweight='bold')
        for c in group.index:
            idx = order.index(c[0])
            sess = group.loc[c]['Session']
            d = (str(date)+'_'+str(sess))
            a = f.add_subplot(len(order), gb.ngroups, idx*gb.ngroups+i+1)
            plot_response(exp_data, rois.iloc[unit].loc[d], d, a)
            a.set_title(c)
            a.set_ylim(-1.5,12.5)
    return f

# plt.tight_layout()

########################################################################################
# # run ttest on trial data (running, tracked ROIs)
# df = pd.DataFrame()
# for date in exp_data.index:
#     ttest_array, resp_cells, percent_response = ttest_responsive(exp_data, rois, date)
#     ttest_res = pd.Series({'ttest': ttest_array,'resp_cells': resp_cells,
#     'percent_response': percent_response}, name=date)
#     df = df.append(ttest_res)
# exp_data = pd.concat([exp_data, df], axis=1)
# f = plot_resp_cells('X', exp_data)
# save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/responsive_cells/'%mouse_id
# save_figs(f, mouse_id, 'all/', '_all', save_path)
