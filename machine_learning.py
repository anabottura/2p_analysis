%load_ext autoreload
%autoreload 2
%matplotlib inline

import seaborn as sns
import pandas as pd
import numpy as np
from my_utils import *
from plotting import *
from sklearn import decomposition, cluster
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

mouse_id = "CTBD7.1d"
output = '/Users/anabottura/Documents/2p_analysis/data/'
filename = output+'%s_exp_data.pickle'%mouse_id
infofile = output+'exp_info.csv'
rois_file = output+'/imaging/processed/%s/rois_mapping/consitent_rois_plane1.csv'%mouse_id
save_path = '/Users/anabottura/Documents/2p_analysis/data/figures/%s/Correlations/'%mouse_id

exp_data, info_df, rois = read_data(mouse_id, filename, infofile, rois_file)

date = '2019/01/31'
w_size = 60

group = info_df.groupby('Date')
g = group.get_group(date)
pre_l = []
post_l = []
vectors = []
for i in g.Session:
    date_sess = date+'_'+str(i)
    # Get X for a specific session and a certain window size
    trials = exp_data.loc[date_sess]['trialArray']
    trials.shape
    stimulus = exp_data.loc[date_sess]['stimulus']
    pre = find_rdm_int(trials, w_size, start=0, stop=int(stimulus))
    post = find_rdm_int(trials, w_size, start=int(stimulus))
    pre = pd.DataFrame(pre.mean(axis=1))
    post = pd.DataFrame(post.mean(axis=1))
    pre = pre[rois[date_sess]]
    post = post[rois[date_sess]]

    #get a compound ID vector
    c_vector = i*np.ones(pre.shape[0])
    vectors.append(c_vector)

    pre_l.append(pre.T.reset_index(drop=True))
    post_l.append(post.T.reset_index(drop=True))

compounds = (pd.DataFrame(np.concatenate(vectors))).T

pre_df = pd.concat(pre_l, axis=1)
post_df = pd.concat(post_l, axis=1)
all_df = (pd.concat([pre_df,post_df], axis=0)).T
all_df.shape

# X = all_df / Y = compounds

# PCA

pca = decomposition.PCA()
pca.fit(all_df)
print(pca.explained_variance_)
sorted = np.sort(pca.explained_variance_ratio_)
sorted.size
plt.plot(np.linspace(0,sorted.size, sorted.size), pca.explained_variance_ratio_)
plt.plot(np.arange(pca.explained_variance_ratio_.shape[0]), pca.explained_variance_)
# As we can see, only the 2 first components are useful
pca.n_components = 10
X_reduced = pca.fit_transform(all_df)
x = pd.DataFrame(X_reduced)
x.shape
X_1 = x[compounds.T.values==1]
X_2 = x[compounds.T.values==2]
X_3 = x[compounds.T.values==3]

plt.scatter(X_1[:][0], X_1[:][1], color='r')
plt.scatter(X_2[:][0], X_2[:][1], color='b')
plt.scatter(X_3[:][0], X_3[:][1], color='g')

plt.scatter(X_1[:][0], X_1[:][2], color='r')
plt.scatter(X_2[:][0], X_2[:][2], color='b')
plt.scatter(X_3[:][0], X_3[:][2], color='g')

plt.scatter(X_1[:][1], X_1[:][2], color='r')
plt.scatter(X_2[:][1], X_2[:][2], color='b')
plt.scatter(X_3[:][1], X_3[:][2], color='g')

# k-means

k_means = cluster.KMeans(n_clusters=4)
k_means.fit(all_df)

print(k_means.labels_)

k_means.labels_.tolist()

unit = 0
all_df[0]
plt.scatter()
plt.scatter(X_1[:][0], X_1[:][1], c=k_means.labels_, cmap=plt.get_cmap('viridis'))

all_df
all_df.shape
compounds.shape

a = all_df.reset_index(drop=True)
# classifier
a.shape
#concat X and Y
concat_df = pd.concat([a, compounds.T], axis=1)
concat_df.shape
# shuffles rows
shuffled = concat_df.sample(frac=1)

shuffledX = shuffled.iloc[:,:-1]
shuffledY = shuffled.iloc[:,-1]
shuffledX.shape
shuffledY.shape

X_train, X_test, y_train, y_test = train_test_split(shuffledX, shuffledY, test_size=0.2, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear').fit(X_train, y_train)
clf.score(X_test, y_test)

# cross validation
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, shuffledX, shuffledY, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
