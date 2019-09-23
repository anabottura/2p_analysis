%load_ext autoreload
%autoreload 2
%matplotlib inline

import seaborn as sns
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

# run ttest on trial data (running, tracked ROIs)
df = pd.DataFrame()
for date in exp_data.index:
    ttest_array, resp_cells, percent_response = ttest_responsive(exp_data, rois, date)
    ttest_res = pd.Series({'ttest': ttest_array,'resp_cells': resp_cells,
    'percent_response': percent_response}, name=date)
    df = df.append(ttest_res)

exp_data = pd.concat([exp_data, df], axis=1)


#save exp_data as a pickled file to output filepath
filename = output+'%s_exp_data.pickle'%mouse_id
with open(filename, 'wb') as f:
    pickle.dump(exp_data, f)

# plot responsive cells
f = plot_resp_cells('B', exp_data)
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/responsive_cells/'%mouse_id
save_figs(f, mouse_id, 'all/', '_B', save_path)
