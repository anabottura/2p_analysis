import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
import scipy.io
from scipy import stats
import os, sys
import tqdm
import pandas as pd

def find_divisors(n):
    factors=[]
    for i in range(1,n+1):
        if n%i==0:
           factors.append(i)
    k = int(len(factors)/2)
    return n/factors[k], factors[k]

def plot_raw(data):
    trace = data
    nb_units = trace.shape[1]

    #plot the data as a heat map
    fig = plt.figure(1)
    ax = plt.imshow(trace.T,aspect="auto")
    plt.xlabel("time (ms)")
    plt.ylabel("Units")
    plt.title('Raw Data')
    return fig

def plot_trialframes(trialArray, stim_start):
    nb_units = trialArray.shape[2]
    my_min = np.min(trialArray.mean(1))
    my_max = np.max(trialArray.mean(1))
    # plot units x trials, averaging frames
    fig2 = plt.figure(2)
    ax2 = plt.imshow(trialArray.mean(1).T, aspect="auto", norm = cm.colors.Normalize(my_min, my_max))
    fig2.colorbar(ax2)
    plt.xlabel("trials")
    plt.ylabel("units")
    plt.title('Average activity per trial')

    # plot units x frames, averaging trials
    fig3 = plt.figure(3)
    x = stim_start*np.ones(nb_units)
    y = np.linspace(0,nb_units,nb_units)
    plt.plot(x, y, '-r', label = 'stimulus')
    my_min = -0.2
    my_max = 1.8
    ax3 = plt.imshow(trialArray.mean(0).T, aspect="auto", norm = cm.colors.Normalize(my_min, my_max))
    fig3.colorbar(ax3)
    plt.legend(loc='lower right')
    plt.xlabel("frames")
    plt.ylabel("units")
    plt.title('Average Trial Activity')
    dic = {'uxt':fig2, 'uxf':fig3}
    return dic

def plot_response(exp_data, unit, date, ax):
    '''
    Function that takes compound, date, unit
    and plots the trials in dF/F x time in axis ax
    '''
    data = exp_data.loc[date]
    trials = data.trialArray[:,:,unit]
    avg = np.mean(trials, axis=0)

    x = np.ones(trials.shape)*(np.linspace(0,trials.shape[1]-1,trials.shape[1]))
    ax.plot(x,trials, '-', color='0.90')
    ax.plot(x[0], avg,'b-')
    ax.axvline(x=exp_data.stimulus.loc[date], color='r')

def plotData(data, trialArray, stim_start):

    fig = plot_raw(data)
    dic = plot_trialframes(trialArray, stim_start)
    dic['RawData']=fig
    return dic

def plot_resp_cells(code, exp_data):
    compounds = exp_data.loc[:,'Compound Code']
    dates = exp_data[compounds==code].index
    p_res_code = exp_data.loc[dates, 'percent_response']
    a = pd.concat([r.unstack() for r in p_res_code])
    list = a.index.levels
    fig_list = {}
    for i in list[0]:
        for k in list[1]:
            trials_cells = a[(i,k)].tolist()
            x = np.linspace(1,len(dates),len(dates))

            fig = plt.figure()
            plt.ylim(0,100)
            plt.title(i+'_'+k)
            plt.xlabel('Sessions')
            plt.ylabel('Responsive Cells (%)')
            if type(trials_cells) == np.float:
                h = 100*trials_cells
            else:
                h = [100*i for i in trials_cells]
            plt.bar(x, height=h)
            fig_list[code+'_'+i+k]= fig
    return fig_list

def plot_tracked(exp_data, rois, start1, start2, sample_size, function):
    a, b = find_divisors(rois.shape[0])
    fig = plt.figure(1, figsize=(8*a, 4*b))
    ttest_df = pd.DataFrame(index=exp_data.index, columns=range(rois.shape[0]))
    pvalues_df = pd.DataFrame(index=exp_data.index, columns=range(rois.shape[0]))
    for u in range(rois.shape[0]):
        df_rn = pd.DataFrame()
        df_pre = pd.DataFrame()
        line_pos = []
        for date in exp_data.index:
            unit = rois.loc[u][date]
            frames = exp_data.Data.loc[date]
            u_frames = frames[:,unit]
            start1_d = start1.loc[date].dropna()
            start2_d = start2.loc[date].dropna()
            df1 = pd.DataFrame([u_frames[int(s):(int(s)+sample_size)] for s in start1_d])
            df2 = pd.DataFrame([u_frames[int(s):(int(s)+sample_size)] for s in start2_d])
            res = stats.ttest_rel(function(df1, axis=1), function(df2, axis=1))
            ttest_df.loc[date][u] = res.statistic
            pvalues_df.loc[date][u] = res.pvalue
            df_rn = df_rn.append(df1)
            df_pre = df_pre.append(df2)
            line_pos.append(start1_d.shape[0])

        df_plot = pd.concat([df_rn,df_pre], axis=1)

        plt.subplot(a,b,u+1)
        x = sample_size*np.ones(df_plot.shape[0]+1)
        y = np.linspace(0,df_plot.shape[0]+1,df_plot.shape[0]+1)
        plt.plot(x, y, '-r', linewidth=1)
        pre_l = 0
        for i, l in enumerate(line_pos):
            ind = l+pre_l
            y = ind*np.ones(df_plot.shape[1]+1)
            x = np.linspace(0,df_plot.shape[1]+1,df_plot.shape[1]+1)
            plt.plot(x, y, '-w', linewidth=0.5)
            pre_l = ind
            if pvalues_df.iloc[i][u]<=0.05:
                plt.text(x[-1]+2,y[i], '*')
        ax = plt.imshow(df_plot, aspect='auto')
        plt.xlabel("frames")
        plt.ylabel("trials")
        plt.title('Tracked Unit %d'%(u+1))

    return fig, ttest_df, pvalues_df

def plot_tracked_unit(unit, exp_data, info_df, rois):
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

def plot_corr(trial_df, title='', b=20):
    corr_matrix = trial_df.corr()
    c = corr_matrix.unstack()
    f = plt.figure(figsize=(10,6))
    f.suptitle('Correlations '+title)
    ax1 = f.add_subplot(121)
    c.plot.hist(bins=b)
    # ax1.set_ylim((0,60))
    f.add_subplot(122)
    plt.imshow(corr_matrix)
    return f, corr_matrix

def plot_cdf(corr_matrix):
    c = corr_matrix.unstack()
    sorted = np.sort(c)
    h, = plt.plot(sorted,np.linspace(0.,1.,sorted.size))
    return h
