# Created by rahman at 12:33 2020-03-09 using PyCharm
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def filter_hashtags_users(DATAPATH, th, city):
    """
    cleans target_hashtags by removing hashtags that are used by less than 2 users
    replaces hahstags by ht_id and saves to idhashtags.csv
    creates entropy for each ht_id and saves to hashtag_id_entropies.csv
    prints std output
    :param DATAPATH:
    :param th: hashtags are too popular if more than th% of users share them
    :param city:
    :return:
    """

    ht = pd.read_csv(DATAPATH + city + ".target_hashtags")
    print ("ht.shape", ht.shape)

    ht["hashtags"] = ht['hashtags'].astype('category')
    ht["ht_id"] = ht["hashtags"].cat.codes
    ht.drop('hashtags', axis=1, inplace=True)

    #arrmult = []
    entarr = []
    gp = ht.groupby('ht_id')

    # cnt_df = gp.size().reset_index(name='sizes')


    # hashtags are too popular if more than th% of users share them
    max_df_ht = th * len(ht.uid.unique())
    print ("max_df_ht", max_df_ht)


    # removing hashtags that are used by less than 2 users and more than th% of users
    for htid, group in gp:

        user_count = len(group['uid'].value_counts().values)

        if user_count > 1 and user_count <= max_df_ht:

            e = entropy(group['uid'].value_counts().values)
            c = len(group)
            entarr.append([htid, e, c])
            #arrmult.append(htid)

    # save entropies of hashtags for other calculations
    entdf = pd.DataFrame(data=entarr, columns=['ht_id', 'entropy', 'counts'])
    sortt = entdf.sort_values(by='entropy')
    sortt.to_csv(DATAPATH + "counts_entropies.csv", index=False)

    # filtered hashtag df
    ht2 = ht[ht.ht_id.isin(entdf.ht_id)]

    print ("after removing too popular and too rare hts", ht2.shape)
    ht2.to_csv(DATAPATH + str(th) + "filtered_hashtags.csv", index=False)

    return entdf, ht2






def make_features(DATAPATH, ht, ent_df, datafile, pairs):
    """
    make 10 features for each friend-stranger pair based on common hashtags
    :param th: entropy filter threshold
    :param ent_df: DF of [entropy, ht_id ]
    :param ht: DF of [ht_id, uid]
    :param datafile: [u1, u2, label, 7 features... ]
    :param pairs: file containing pairs shared among all thresholds
    :return: dataset
    """

    print ("pairs.shape", pairs.shape)


    countdf = ht.groupby('ht_id')['uid'].count().reset_index(name='counts')

    # htids is a dict of uid as keys and array of htids as values, containing hashtag ids shared by each user id
    gp = ht.groupby('uid')['ht_id']
    htids = {}
    for userid, val in gp:
        htids[userid] = val.values



    dataset = pairs #pd.read_csv(DATAPATH+ datafile)
    count=0

    for i in dataset.index:

        try:

            u1, u2 = dataset.loc[i,'u1'], dataset.loc[i,'u2']

            overlap = list(set(htids[u1]).intersection(set(htids[u2])))

            if len(overlap) > 0:

                count+=1

                # value counts of all ht_ids for a user
                v2 = gp.get_group(u2).value_counts()[overlap]
                v1 = gp.get_group(u1).value_counts()[overlap]

                # aa_ent, sum of the reciprocals of entropies of each common hashtag
                aa = pd.np.sum(pd.np.reciprocal(ent_df[ent_df.ht_id.isin(overlap)].entropy))

                unionn=set(htids[u1]).union(set(htids[u2]))

                dataset.loc[i, 'cos_dis']       = cosine(v1, v2)                # cosine similarity between the user-hashtag count vectors

                dataset.loc[i, 'dot_pro']       = pd.np.dot(v1, v2)             # dot product of) number of times the common hashtags were shared by both users

                dataset.loc[i, 'common_h']      = len(overlap)                  # number of hashtags shared by both users

                dataset.loc[i, 'overlap_h']     = len(overlap)*1.0/len(unionn)  # and fraction

                dataset.loc[i, 'min_ent']       = ent_df[ent_df.ht_id.isin(overlap)].entropy.min() # minimum among the entropies of all common hashtags

                dataset.loc[i, 'aa_ent']        = aa                            # sum of the reciprocals of entropies of each common hashtag

                dataset.loc[i, 'count_aa_ent']  = aa * 1.0 / len(unionn)        # aa_ent weighted by the number of  hashtags shared by both users

                dataset.loc[i, 'big_aa_ent']    = aa * 1.0 / pd.np.sum(pd.np.reciprocal(ent_df[ent_df.ht_id.isin(unionn)].entropy))  # aa weighted by the sum of the reciprocals of entropies of hashtags shared by both users.

                dataset.loc[i, 'min_h']         = countdf[countdf.ht_id.isin(overlap)].counts.min()                                  # minimum of the total number of times all users shared a hashtag among the set of common hashtags between u1 and u2

                dataset.loc[i, 'aa_h']          = pd.np.sum(pd.np.reciprocal(pd.np.log(countdf[countdf.ht_id.isin(overlap)].counts)))#sum of the reciprocals of the logarithm of nhi
                                                                                                                                     # which is the number of times a hashtag hi common between u and v was shared by all users.
        except:

            continue


    print ("count", count)

    print ("before dropna dataset[dataset.label==0].shape, dataset[dataset.label==1].shape", \
           dataset[dataset.label == 0].shape, dataset[dataset.label == 1].shape)

    dataset.dropna(inplace=True)

    print ("afer dropna dataset[dataset.label==0].shape, dataset[dataset.label==1].shape",   \
           dataset[dataset.label == 0].shape, dataset[dataset.label == 1].shape)



    """ #  to make both classes equally represented but we use AUC in the paper so we dont need 
    class1 = dataset[dataset.label == 1]
    class0 = dataset[dataset.label == 0]
    bigClass = class1 if len(class1) >= len(class0) else class0
    smallClass = class1 if len(class1) < len(class0) else class0

    #try:
    sampledClass = bigClass.sample(len(smallClass))
    dataset = sampledClass.append(smallClass, ignore_index=True)
    print "dataset.shape", dataset.shape"""

    dataset.to_csv(DATAPATH + datafile, index=False)

    return dataset




def score(dataset,  classifiers, UNSUP):
    """
    trains classifier and calculates 'accuracy', 'roc_auc', 'average_precision' with n_splits=3, n_repeats=10
    prints std output

    :param dataset: header of the form [u1,u2,label,features 1- 7 ]
    :param classifiers: {'RF':(RandomForestClassifier, {"n_estimators": 101,  "max_depth": 10}),
                  'AB':(AdaBoostClassifier, {"n_estimators": 101}),
             #'GBM': (GradientBoostingClassifier,{'n_estimators':100, 'max_depth': 3}),
             #'LR_SAG_L2penalty':(LogisticRegression, {'solver': 'sag'}),
             #'LR_liblinear_L2penalty': (LogisticRegression, {'solver': 'liblinear', 'penalty': 'l2'})}

    :param UNSUP: Boolean if False, attack is supervised only i.e. features are embedding from word2vec
    :return: None
    """

    result, names = [], []


    if UNSUP:

        for col in dataset.columns[3:]:

            val = roc_auc_score(dataset.label, dataset[col])
            roc = 1 - val if val < 0.5 else val

            print ("roc_auc_score ", col,  roc)

            names.append(col)
            result.append(roc)



    for cname, classifier in classifiers.items():

        print ("scoring ", len(dataset.columns[3:]), "features with ", cname)

        clf = classifier[0](**classifier[1])

        dataset = dataset.sample(frac=1).reset_index(drop=True)

        X = dataset.iloc[:, 3:].values
        Y = dataset.iloc[:, 2].values

        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)

        for scorer in ['roc_auc']:#, 'accuracy', 'average_precision']:

            scores = cross_val_score(clf, X, Y, scoring=scorer, cv=rskf)

            print(scorer, " %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

            result.append(scores.mean())
            names.append(cname)

    return result, names
