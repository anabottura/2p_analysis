import pickle
import pandas as pd
import numpy as np
from my_utils import *

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)

date ='2019/01/31_1'

licks = exp_data.Licks.loc[date]
activity = exp_data.Data.loc[date]
activity.shape
avg_act = np.mean(activity, axis=1)
avg_act.shape
licks = [int(l) for l in licks]
plt.plot(np.linspace(0,len(licks), len(licks)),avg_act[licks])
plt.show()

plt.clf()

window = 90
after_licks = []
for l in licks:
    df = pd.DataFrame(activity[l:l+window])
    df = df.mean(axis=1)
    df.name = l
    after_licks.append(df)

df_licks = (pd.concat(after_licks, axis=1)).T

plt.figure(figsize=(6,4))
for i, e in enumerate(df_licks.index):
    plt.plot(np.linspace(0,window, window), df_licks.iloc[i])
plt.tight_layout()
plt.show()

avg = df_licks.mean(axis=0)

f = plt.figure(figsize=(6,4))
plt.plot(np.linspace(0,window, window), avg)
plt.ylim(-0.1,0.4)
plt.tight_layout()
plt.show()
