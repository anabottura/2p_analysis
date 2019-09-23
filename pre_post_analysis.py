%load_ext autoreload
%autoreload 2
%matplotlib inline

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


sample_size = int(exp_data.stimulus[0])
windows = exp_data.trial_limit
stimulus = exp_data.stimulus
start1=pd.DataFrame([pd.Series(r[0]) for r in windows])
start1.index = windows.index
start2 = pd.DataFrame([pd.Series(r[0]).add(sample_size*np.ones(r[0].shape)) for r in windows])
start2.index = windows.index

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.trapz)

spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
name = 'pre_post_area'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.svg'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.mean)

spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
name = 'pre_post_mean'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.svg'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.max)

spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
name = 'pre_post_max'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.svg'%(mouse_id, name), dpi=300)
f.clf()

f, ttest_df, pvalues_df = plot_tracked(exp_data, rois, start1, start2, sample_size, np.median)

spath = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id
name = 'pre_post_median'
if not os.path.exists(spath):
    os.makedirs(spath)
f.savefig(spath+'%s_%s.svg'%(mouse_id, name), dpi=300)
f.clf()
