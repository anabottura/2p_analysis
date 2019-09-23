import pickle
import pandas as pd
import numpy as np
from my_utils import *
from plotting import *

mouse_id = "CTBD11.1i"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)

# to get random intervals outside of the trial interval
sample_size = 60 #size of the window I will take in frames

windows = exp_data.trial_limit
stimulus = exp_data.stimulus
sample_size_df = stimulus.copy()
sample_size_df[:] = sample_size #create a df with the value of the window
#reshape the trial limits to have it in a dataframe
a=pd.DataFrame([np.array([r[0]]) for r in windows], index=windows.index, columns=['trial_start'])
b=pd.DataFrame([np.array([r[1]]) for r in windows], index=windows.index, columns=['trial_end'])
windows = pd.concat([a,b],axis=1)

to_sub = sample_size_df*[np.ones(r[0].shape) for r in windows['trial_start']]
w_end = windows.trial_start.subtract(to_sub)
w_end = w_end.subtract(to_sub)
w_start = windows.trial_start.add(to_sub)
b_window = pd.concat([w_start,w_end],axis=1)

random_ints = pd.DataFrame(index=b_window.index, columns=['start_random', 'start_pre'])
for date in b_window.index:
    a = b_window.loc[date]
    b = windows.trial_start[date]
    start_rn = a.iloc[0][0][:-1]
    end_rn = a.iloc[1][0][1:]
    n_start_rn = start_rn[end_rn-start_rn>0]
    n_end_rn = end_rn[end_rn-start_rn>0]
    start_pre = b[0][1:]
    n_start_pre = start_pre[end_rn-start_rn>0]
    br = [np.random.randint(s, e) for s,e in zip(n_start_rn,n_end_rn)]
    random_ints.loc[date]['start_random'] = pd.Series(br, name=date)
    random_ints.loc[date]['start_pre'] = pd.Series(n_start_pre, name=date)

start1 = random_ints.start_random
start2 = random_ints.start_pre

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.trapz)

spath = output+'/figures/%s/'%mouse_id
name = 'rdm_pre_area'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.mean)

spath = output+'/figures/%s/'%mouse_id
name = 'rdm_pre_mean'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.median)

spath = output+'/figures/%s/'%mouse_id
name = 'rdm_pre_median'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.max)

spath = output+'/figures/%s/'%mouse_id
name = 'rdm_pre_max'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
f.clf()
