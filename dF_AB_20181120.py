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
sys.path.append(os.path.expanduser('/Users/anacarolinabotturabarros/Documents/bitbucket/pynalysis'))
from pynalysis import utils

sys.path.append(os.path.expanduser('/Users/anacarolinabotturabarros/Documents/bitbucket/tensortools-master/'))
import tensortools as tt

# Import local OASIS installation
sys.path.append('/Users/anacarolinabotturabarros/Documents/bitbucket/OASIS')
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis import oasisAR1, oasisAR2

# Set file-name and path to analyze
stringid = "CBCB1110.2d"
filename = os.path.expanduser('/Users/anacarolinabotturabarros/data/processed/imaging/%s.mat'%stringid)

# Load data struct
dat = utils.load_mat_file(filename)

# Print contents
print("subject: %s"%stringid)
utils.print_file_content(dat)

area_name = "area1"
expspecifier = "date_2018_06_28/%s"%area_name
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

licks = sessbehav['lick_event'].value
startTrial = sessbehav['lick_wait'].value #contains the start of the trials when the program is expecting a lick to give a reward

#if the last window waiting for a lick does not have a lick after then remove the last 'lick_wait'
if startTrial[-1]>licks[-1]:
    startTrial = startTrial[:-1]

#finds the frames when the animal licks to get reward
stimStart = []
for trial in startTrial:
    tempAr=licks[licks>=trial]
    stimStart.append(tempAr[0]) # takes the first lick after 'lick_wait' as the lick that gets the reward

stimStart = np.array(stimStart)
data_raw=plane1['fluoresence_corrected'].value

indexnan = np.isnan(data_raw.mean(0))

data = data_raw[:,indexnan==False]

np.isnan(data.mean())

data.shape



plt.figure
ax = plt.imshow(data.T,aspect="auto")

unit = 1
trace = data
nb_units = trace.shape[1]
before = 0.5 #second
after = 1.5 #second
bFrames = int(before*frate)
aFrames = int(after*frate)

trialArray = np.zeros((len(stimStart), bFrames+aFrames, nb_units))
for i, t in enumerate(stimStart):
    trialArray[i,:] = np.array(trace[int(t-bFrames):int(t+aFrames)])


trialArray.shape

# plt.imshow(trialArray.mean(1).T, aspect="auto")
# plt.xlabel("trials")
# plt.ylabel("units")
#
# plt.imshow(trialArray.mean(0).T, aspect="auto")
# plt.xlabel("frames")
# plt.ylabel("units")

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
