import pickle
import pandas as pd
import numpy as np
from my_utils import *

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id

# Load exp. data file
with open(filename, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    exp_data = pickle.load(f)

# Read the information about the experiment
info_df = read_exp_info(infofile, mouse_id)
info_df = info_df[2:] #remove the pretraining Sessions --- need to change this

# Read csv file with tracked ROIs
names = []
for s, name in enumerate(info_df.Date.values):
    names.append(name+'_'+str(info_df.Session.values[s]))
rois = pd.read_csv(rois_file, header=None, names=names)

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
f.savefig(spath+'%s_%s.pdf'%(mouse_id, name), dpi=300)
f.clf()
