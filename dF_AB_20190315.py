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

# Import local tools
sys.path.append(os.path.expanduser('/Users/anabottura/github/pynalysis/'))
from pynalysis import utils

def load_im_struc(mouse_id, date, session, filepath):
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

def preprocess(mouse_id, date, session, filepath, trial_window=[1,3]):

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
        tempAr=tempAr[tempAr<iti_start[t]]
        if tempAr.any() == False:
            continue
        # add that time to the stimStart list
        stimStart.append(tempAr[0]) # takes the first lick after 'trial_start' as the lick that gets the reward

    stimStart = np.array(stimStart)

    data_raw=plane1['fluoresence_corrected'].value

    #check if there are nan values
    indexnan = np.isnan(data_raw.mean(0))
    data = data_raw[:,indexnan==False]
    np.isnan(data.mean())

    # data.shape

    #plot the data as a heat map
    fig = plt.figure(1)
    ax = plt.imshow(data.T,aspect="auto")
    plt.xlabel("time (ms)")
    plt.ylabel("Units")
    plt.title('Raw Data')

    trace = data
    nb_units = trace.shape[1]
    bFrames = int(trial_window[0]*frate)
    aFrames = int(trial_window[1]*frate)

    # create an array of trials x frames x units
    trialArray = np.zeros((len(stimStart), bFrames+aFrames, nb_units))
    trial_window = np.zeros((2,len(stimStart)), dtype=int)
    for i, t in enumerate(stimStart):
        startOfTrial = int(t-bFrames)
        endOfTrial = int(t+aFrames)
        if startOfTrial < 0:
            continue
        if endOfTrial > trace.shape[0]:
            continue
        trial_window[0,i]=startOfTrial
        trial_window[1,i]=endOfTrial
        trialArray[i,:] = np.array(trace[startOfTrial:endOfTrial])

    trialArray.shape

    # plot units x trials, averaging frames
    fig2 = plt.figure(2)
    ax2 = plt.imshow(trialArray.mean(1).T, aspect="auto")
    plt.xlabel("trials")
    plt.ylabel("units")
    plt.title('Average activity per trial')

    # plot units x frames, averaging trials
    fig3 = plt.figure(3)
    x = bFrames*np.ones(nb_units)
    y = np.linspace(0,nb_units,nb_units)
    plt.plot(x, y, '-r', label = 'stimulus')
    ax3 = plt.imshow(trialArray.mean(0).T, aspect="auto")
    plt.legend(loc='lower right')
    plt.xlabel("frames")
    plt.ylabel("units")
    plt.title('Average Trial Activity')

    fig = {'RawData':fig, 'uxt':fig2, 'uxf':fig3}
    return data, trialArray, trial_window, fig

def save_figs(fig_list, mouse_id, date, session, save_path):
    for name, f in fig_list.items():
        spath=((save_path+'%s/')%name)
        if not os.path.exists(spath):
            os.makedirs(spath)
        f.savefig(spath+'%s_%s_Session%s_%s.png'%(mouse_id, date.replace('/','_'), session, name), dpi=300)
        f.clf()


def find_run(window, mouse_id, date, session, filepath, threshold=40):

    area, plane1, sessbehav = load_im_struc(mouse_id, date, session, filepath)

    speed = sessbehav['velocity'].value
    tspeed = np.zeros(plane1['fluoresence_corrected'].shape[0])
    tspeed[:speed.shape[1]] = speed
    speed = tspeed

    print('Average speed: %f'%(np.mean(speed)))

    running_t = np.zeros(len(window.T))
    trial_n_frames = (trial_window[1]-trial_window[0])[0]
    trial_speed = np.zeros((len(window.T),trial_n_frames))
    for i, t in enumerate(window.T):
        trial_speed[i,:] = speed[t[0]:t[1]]
        if np.mean(trial_speed[i,:]) > threshold:
            running_t[i] = 1
    return running_t, trial_speed

infofile = '/Users/anabottura/Documents/2p_analysis/data/exp_info.csv'
mouse_id = "CTBD7.1d"
filepath = '/Users/anabottura/Documents/2p_analysis/data/pipeline_output/imaging/'
rois_file = '/Users/anabottura/Documents/2p_analysis/data/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/'%mouse_id

info_df = read_exp_info(infofile, mouse_id)
info_df = info_df[2:]

# read csv file with tracked ROIs
names = []
for s, name in enumerate(info_df.Date.values):
    names.append(name+'_'+str(info_df.Session.values[s]))
rois = pd.read_csv(rois_file, header=None, names=names)


session_info = info_df.iloc[0]
session_info
date = session_info.Date
session = "area"+str(session_info.Session)


data, trials, trial_window, fig_list = preprocess(mouse_id, date, session, filepath)
# save_figs(fig_list, mouse_id, date, str(session_info.Session), save_path)

run_trials, trial_speed = find_run(trial_window, mouse_id, date, session, filepath)


for i in range(info_df.shape[0]):
    session_info = info_df.iloc[i]
    date = session_info.Date
    session = "area"+str(session_info.Session)

    data, trials, fig_list = preprocess(mouse_id, date, session, filepath)
    save_figs(fig_list, mouse_id, date, str(session_info.Session), save_path)
