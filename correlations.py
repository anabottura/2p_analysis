%load_ext autoreload
%autoreload 2
%matplotlib inline

import seaborn as sns
import pandas as pd
import numpy as np
from my_utils import *
from plotting import *

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/Correlations/'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)

code = 'AX'
corr_all, fig_all = get_corr_plots(exp_data, code, name= 'of all cells')
corr_resp, fig_resp = get_corr_plots(exp_data, code,name= 'of responsive cells', cells = exp_data.resp_cells)

cdf = {'All': fig_all, 'Resp': fig_resp}
save_figs(cdf, mouse_id, 'alldays', code+'_cdf_', save_path)

dic_list = []
for d in corr_all.Figures.index:
    dic = {}
    dic.update({'All':corr_all.Figures.loc[d]})
    dic.update({'Resp':corr_resp.Figures.loc[d]})
    dic_list.append(dic)

for i, dic in enumerate(dic_list):
    save_figs(dic, mouse_id, corr_all.Figures.index[i], code, save_path)

# def get_corr(code):
#     corr_all, fig_all = get_corr_plots(exp_data, code, name= 'of all cells')
#     corr_resp, fig_resp = get_corr_plots(exp_data, code,name= 'of responsive cells', cells = exp_data.resp_cells)
#     return corr_all, fig_all, corr_resp, fig_resp

def get_corr_plots(exp_data, code, name='', cells=pd.DataFrame(), cell_id='AllCells', trial_id='AllTrials'):
    sessions = exp_data.trialArray.loc[exp_data['Compound Code']==code]
    corrs = pd.DataFrame(columns=['Correlations', 'Figures'], index=sessions.index)
    for date in sessions.index:
        trialAvg = np.mean(exp_data.trialArray[date], axis=0)
        if cells.empty:
            selected_trials = pd.DataFrame(trialAvg)
        else:
            resp_cells = cells.loc[date]
            resp_cells = resp_cells.loc[cell_id][trial_id]
            r_cells = [i for (i,e) in enumerate(resp_cells) if e == True]
            t_cells = trialAvg[:,r_cells]
            selected_trials = pd.DataFrame(t_cells)
        ti = name+': '+code+' - '+date[:-2]
        f2, c2 = plot_corr(selected_trials, title=ti)
        corrs.loc[date]['Figures'] = f2
        corrs.loc[date]['Correlations'] = c2

    fig = plt.figure(figsize=(8,8))
    fig.suptitle('Cumulative distribution '+name+': '+code)
    handles = []
    labels = []
    for c in corrs.index:
        u_c = corrs.loc[c]['Correlations']
        h = plot_cdf(u_c)
        handles.append(h)
        labels.append(c[:-2])
    fig.legend(handles = handles, labels = labels, loc='lower right')

    return corrs, fig
