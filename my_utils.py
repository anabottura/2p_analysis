# Import generic libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io
from scipy import stats
import os, sys
import tqdm
import pandas as pd
import pickle

sys.path.append(os.path.expanduser('/Users/anabottura/github/pynalysis/'))
from pynalysis import utils

# create Dataframe with the experimental info
def read_exp_info(file, MouseID):
    '''Read a csv file with information about the experiment and returns a dataFrame with data from 'MouseID'.
    file: file path with filename
    MouseID: mouse identification
    Returns
    df: a Pandas Dataframe with the info from the specified Mouse
    '''
    df = pd.read_csv(file)
    df = df.loc[df['Mouse ID'] == MouseID]

    return df

def load_im_struc(mouse_id, date, session, filepath):
    ''' Loads a matlab structure that results from running the allignment pipeline.
    mouse_id: mouse identification
    date: date of experiment in the form of YYYY/MM/DD
    session: session or area to find files
    filepath: file path wehre MATLAB structure is saved

    Returns
    area: full structure
    plane1: imaging structure
    sessbehav: behavioural structure
    '''
    # Set file-name and path to analyze
    filename = os.path.expanduser((filepath+'%s.mat')%mouse_id)
    # Load data struct
    dat = utils.load_mat_file(filename)

    # Print contents
    print("subject: %s"%mouse_id)
    # utils.print_file_content(dat)

    expspecifier = ('date_'+date.replace('/','_')+'/%s')%session
    path="imaging/%s"%expspecifier
    print(path)
    area=dat.get(path)
    utils.get_hdf5group_keys(area)

    plane1=area['plane1']
    sessbehav=area['session_behaviour']

    print('Data shape: '+str(plane1['fluoresence_corrected'].shape))

    print(utils.get_hdf5group_keys(sessbehav))
    print(utils.get_hdf5group_keys(plane1))
    return area, plane1, sessbehav

def preprocess(mouse_id, date, session, filepath, trial_window=[3,3]):

    area, plane1, sessbehav = load_im_struc(mouse_id, date, session, filepath)

    # Extract frame rate from struct
    # print(utils.get_hdf5group_keys(area['plane1']))
    frate = area['plane1/fRate'][0,0]
    print("Frame rate %f"%frate)

    licks1 = sessbehav['lick1_event'].value
    licks2 = sessbehav['lick2_event'].value
    if len(licks1)>len(licks2):
        licks = licks1
    else:
        licks = licks2
    startTrial = sessbehav['trial_start'].value #contains the start of the trials when the program is expecting a lick to give a reward
    iti_start = sessbehav['iti_start'].value
    # print('trials: '+str(startTrial))
    # print('licks: '+str(licks))
    #if the last window waiting for a lick does not have a lick after then remove the last trial start
    startTrial = startTrial[:len(iti_start)]
    if startTrial[-1]>licks[-1]:
        startTrial = startTrial[:-1]

        #finds the frames when the animal licks for the first time after reward is delivered
    stimStart = []

    # print('licks: '+str(len(licks)))
    # print('iti_start: '+str(len(iti_start)))
    # print('startTrial: '+str(len(startTrial)))
    for t, trial in enumerate(startTrial):
        # find the lick just after reward delivery
        tempAr=licks[licks>=trial]
        # print(tempAr)
        tempAr=tempAr[tempAr<iti_start[t]]
        if tempAr.any() == False:
            continue
        # add that time to the stimStart list
        stimStart.append(tempAr[0]) # takes the first lick after 'trial_start' as the lick that gets the reward

    stimStart = np.array(stimStart)
    data_raw = plane1['fluoresence_corrected'].value
    #check if there are nan values
    indexnan = np.isnan(data_raw.mean(0))
    data = data_raw[:,indexnan==False]

    trace = data
    nb_units = trace.shape[1]
    bFrames = int(trial_window[0]*frate)
    aFrames = int(trial_window[1]*frate)
    # create an array of trials x frames x units
    trialArray = []
    trials_windows = []
    for i, t in enumerate(stimStart):
        startOfTrial = int(t-bFrames)
        endOfTrial = int(t+aFrames)
        if startOfTrial < 0:
            continue
        if endOfTrial > trace.shape[0]:
            continue
        trials_windows.append(np.array([startOfTrial, endOfTrial], dtype=int))
        trialArray.append(np.array(trace[startOfTrial:endOfTrial]))

    trials_windows = np.array(trials_windows).T
    trialArray = np.array(trialArray)

    return data, trialArray, trials_windows, bFrames, licks

def save_figs(fig_list, mouse_id, date, session, save_path):
    '''
    Gets a dictionary of figures from a certain mouse, date and session
    and saves each figure to file on save_path.
    '''
    for name, f in fig_list.items():
        spath=((save_path+'%s/')%name)
        if not os.path.exists(spath):
            os.makedirs(spath)
        f.savefig(spath+'%s_%s_Session%s_%s.png'%(mouse_id, date.replace('/','_'), session, name), dpi=300)
        f.clf()

def find_run(window, mouse_id, date, session, filepath, threshold=40):
    '''
    Given a array of trial windows (window), a mouse, date and session,
    finds the trials where the average speed is above threshold
    '''
    #load structure
    area, plane1, sessbehav = load_im_struc(mouse_id, date, session, filepath)

    speed = sessbehav['velocity'].value #gets the speed from the structure

    # since speed is acquired in a slightly different frame rate need to add missing frames
    tspeed = np.zeros(plane1['fluoresence_corrected'].shape[0])
    tspeed[:speed.shape[1]] = speed
    speed = tspeed

    print('Average speed: %f'%(np.mean(speed)))

    running_t = np.zeros(len(window.T)) #array that will contain running (1) and no-running (0) trials
    trial_n_frames = (window[1]-window[0])[0]
    trial_speed = np.zeros((len(window.T),trial_n_frames)) #array for the speed split into trials
    for i, t in enumerate(window.T):
        trial_speed[i,:] = speed[t[0]:t[1]]
        if np.mean(trial_speed[i,:]) > threshold:
            running_t[i] = 1
    return running_t, trial_speed

def ttest_responsive(exp_data, rois, date):
    trials = exp_data.loc[date, 'trialArray']
    stim = exp_data.loc[date, 'stimulus']
    run_t = exp_data.loc[date, 'running_t']
    sess_rois = rois[date]
    run_trials = trials[run_t==1,:,:]
    trials_tracked = trials[:,:,sess_rois]
    run_trials_tracked = run_trials[:,:,sess_rois]
    pre_trialCells = [trials[:,:int(stim),:], run_trials[:,:int(stim),:], trials_tracked[:,:int(stim),:], run_trials_tracked[:,:int(stim),:]]
    post_trialCells = [trials[:,int(stim):,:], run_trials[:,int(stim):,:], trials_tracked[:,int(stim):,:], run_trials_tracked[:,int(stim):,:]]
    # average over frames
    for i, array in enumerate(pre_trialCells):
        pre_trialCells[i] = np.mean(array, axis=1)
    for i, array in enumerate(post_trialCells):
        post_trialCells[i] = np.mean(array, axis=1)
    # do paired samples t-test on trials
    ttest_array = []
    for i, array in enumerate(pre_trialCells):
        ttest = stats.ttest_rel(pre_trialCells[i], post_trialCells[i])
        ttest_array.append(ttest)
    #find 'responsive cells'
    resp_cells = []
    for i, array in enumerate(ttest_array):
        res = array.pvalue < 0.05
        resp_cells.append(res)
    percent_response = []
    for i, array in enumerate(resp_cells):
        pr = np.count_nonzero(array)/array.shape[0]
        percent_response.append(pr)

    ttest_array = pd.DataFrame([[ttest_array[0],ttest_array[1]],[ttest_array[2],ttest_array[3]]], index=['AllCells','Tracked'], columns=['AllTrials', 'Running'])
    resp_cells = pd.DataFrame([[resp_cells[0],resp_cells[1]],[resp_cells[2],resp_cells[3]]], index=['AllCells','Tracked'], columns=['AllTrials', 'Running'])
    percent_response = pd.DataFrame([[percent_response[0],percent_response[1]],[percent_response[2],percent_response[3]]], index=['AllCells','Tracked'], columns=['AllTrials', 'Running'])

    return ttest_array, resp_cells, percent_response

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

def read_data(mouse_id, pickle_file, infofile, rois_file):
    # Load exp. data file
    with open(pickle_file, 'rb') as f:
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

    return exp_data, info_df, rois

def find_rdm_int(trials, size, start=0, stop=''):
    if stop == '':
        stop = trials.shape[1]
    end_w = stop-size
    n_start = np.random.randint(start, end_w)
    trial_slice = trials[:,n_start:n_start+size,:]
    return trial_slice

#need to create a function with this... not sure if it is somethin necessary
#plot peri stim activity for responsive cells, tracked
# compounds = exp_data.loc[:,'Compound Code']
# slice = exp_data[compounds==code]
# trial_slice = slice.loc[:, 'trialArray']
# cells_slice = slice.loc[:, 'resp_cells']
# trials_resp_cells = []
# for d, date in enumerate(trial_slice.index):
#     trials_date = trial_slice.loc[date]
#     trials_date = trials_date[:,:,rois[date]]
#     trials_date = trials_date[:,:,cells_slice[date].loc['Tracked', 'AllTrials']]
#     print(trials_date.shape)
#     trials_resp_cells.append(trials_date)
#     fig1, fig2 = plot_trialframes(trials_date, stim_start)
#     fig_list = {'uxt':fig1, 'uxf':fig2}
#     # save_figs(fig_list, mouse_id, date, code, save_path)
