%load_ext autoreload
%autoreload 2
%matplotlib inline

#test
# Import generic libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io
from scipy import stats
import os, sys
import tqdm

results = dict()

# Import local tools
sys.path.append(os.path.expanduser('/Users/anabottura/github/pynalysis/'))
from pynalysis import utils

# sys.path.append(os.path.expanduser('/Users/anabottura/github/tensortools/tensortools/'))
# import tensortools as tt

# # Import local OASIS installation
# sys.path.append('/Users/anabottura/github/OASIS/oasis')
# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
# from oasis import oasisAR1, oasisAR2

# Set file-name and path to analyze
stringid = "CTBD7.1d"
filename = os.path.expanduser('/Users/anabottura/Documents/2p_analysis/data/pipeline_output/imaging/%s.mat'%stringid)

# Load data struct
dat = utils.load_mat_file(filename)

# Print contents
print("subject: %s"%stringid)
utils.print_file_content(dat)

area_name = "area2"
expspecifier = "date_2019_01_31/%s"%area_name
path="imaging/%s"%expspecifier
area=dat.get(path)
utils.get_hdf5group_keys(area)

plane1=area['plane1']
# plane2=area['plane2']
sessbehav=area['session_behaviour']

print(plane1['fluoresence_corrected'].shape)
# print(plane2['fluoresence_corrected'].shape[1])

print(utils.get_hdf5group_keys(sessbehav))
print(utils.get_hdf5group_keys(plane1))

# Extract frame rate from struct
# print(utils.get_hdf5group_keys(area['plane1']))
frate = area['plane1/fRate'][0,0]
print("Frame rate %f"%frate)

licks = sessbehav['lick2_event'].value
startTrial = sessbehav['trial_start'].value #contains the start of the trials when the program is expecting a lick to give a reward
iti_start = sessbehav['iti_start'].value
licks
startTrial

sessbehav['trial_start']
#if the last window waiting for a lick does not have a lick after then remove the last trial start

if startTrial[-1]>licks[-1]:
    startTrial = startTrial[:-1]

#finds the frames when the animal licks for the first time after reward is delivered
stimStart = []
startTrial
for t, trial in enumerate(startTrial):
    # find the lick just after reward delivery
    tempAr=licks[licks>=trial]
    tempAr=tempAr[tempAr<iti_start[t]]
    # add that time to the stimStart list
    stimStart.append(tempAr[0]) # takes the first lick after 'trial_start' as the lick that gets the reward
# #finds the frames when the animal licks to get reward
# stimStart = []
# for trial in startTrial:
#     tempAr=licks[licks>=trial]
#     stimStart.append(tempAr[0]) # takes the first lick after 'lick_wait' as the lick that gets the reward
stimStart = np.array(stimStart)

data_raw=plane1['fluoresence_corrected'].value

#check if there are nan values
indexnan = np.isnan(data_raw.mean(0))
data = data_raw[:,indexnan==False]
np.isnan(data.mean())

data.shape

#plot the data as a heat map
plt.figure
ax = plt.imshow(data.T,aspect="auto")
plt.xlabel("time (ms)")
plt.ylabel("units")

unit = 1
trace = data
nb_units = trace.shape[1]
before = 1 #second
after = 3 #second
bFrames = int(before*frate)
aFrames = int(after*frate)

stimStart

bFrames
# create an array of trials x frames x units
trialArray = np.zeros((len(stimStart), bFrames+aFrames, nb_units))
for i, t in enumerate(stimStart):
    startOfTrial = int(t-bFrames)
    endOfTrial = int(t+aFrames)
    if startOfTrial < 0:
        continue
    if endOfTrial > trace.shape[0]:
        continue
    trialArray[i,:] = np.array(trace[startOfTrial:endOfTrial])

trialArray.shape

# plot units x trials, averaging frames
plt.imshow(trialArray.mean(1).T, aspect="auto")
plt.xlabel("trials")
plt.ylabel("units")

# plot units x frames, averaging trials
x = bFrames*np.ones(nb_units)
y = np.linspace(0,nb_units,nb_units)
plt.plot(x, y, '-r', label = 'stimulus')
plt.imshow(trialArray.mean(0).T, aspect="auto")
plt.legend(loc='lower right')
plt.xlabel("frames")
plt.ylabel("units")



#creates the trial_type array where we identify which trials correspond to what stimulus
stimID = ['A', 'Y']
stimLastFrame = [30000, 60000]
trial_type = np.ones(stimStart.shape)
firstStim = stimStart<stimLastFrame[0]
trial_type[firstStim == False] = 0

# plt.imshow(trialArray[trial_type==0].mean(1).T, aspect="auto")
# plt.xlabel("trials")
# plt.ylabel("units")
#
# plt.imshow(trialArray[trial_type==1].mean(1).T, aspect="auto")
# plt.xlabel("trials")
# plt.ylabel("units")

#Find the pre and post frames for each trial
trialArray_pre = trialArray[:, 0:bFrames, :]
trialArray_post = trialArray[:, bFrames:, :]

# average over frames
av_pre = np.mean(trialArray_pre, axis=1)
av_post = np.mean(trialArray_post, axis=1)

# do paired samples t-test on trials
ttest = stats.ttest_rel(av_pre, av_post)
res = ttest.pvalue < 0.05
percent_response = np.count_nonzero(res)/trialArray.shape[2]
percent_response

# get the trials of each type
array_stim1_pre = av_pre[trial_type==1]
array_stim1_post = av_post[trial_type==1]
array_stim2_pre = av_pre[trial_type==0]
array_stim2_post = av_post[trial_type==0]

#for stim1
ttest = stats.ttest_rel(array_stim1_pre, array_stim1_post)
res = ttest.pvalue < 0.05
percent_response = np.count_nonzero(res)/trialArray.shape[2]
percent_response

#for stim2
ttest = stats.ttest_rel(array_stim2_pre, array_stim2_post)
res = ttest.pvalue < 0.05
percent_response = np.count_nonzero(res)/trialArray.shape[2]
percent_response
