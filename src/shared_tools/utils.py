# Created by rahman at 14:51 2020-03-05 using PyCharm

import os
import random

import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

city = 'la' #'ny'
DATAPATH = '../data/' + city + "/"




classifiers = {
                'RF':(RandomForestClassifier, {"n_estimators": 101, "max_depth": 10}),
                'GBM': (GradientBoostingClassifier,{'n_estimators':100, 'max_depth': 3}),
                'AB':(AdaBoostClassifier, {"n_estimators": 101}),
                'LR_SAG_L2penalty':(LogisticRegression, {'solver': 'sag'}),
                'LR_liblinear_L2penalty': (LogisticRegression, {'solver': 'liblinear', 'penalty': 'l2'})}



def folder_setup(city):
    '''setup folders for each city
    Args:
        city: city
    Returns:
    '''
    if not os.path.exists('data/'+city):
        os.mkdir('data/'+city)

    if not os.path.exists('data/'+city+'/emb/'):
        os.mkdir('data/'+city+'/emb/')

    if not os.path.exists('data/'+city+'/feature/'):
        os.mkdir('data/'+city+'/feature/')

    #os.mkdir('data/'+city+'/process/')
    if not os.path.exists('data/'+city+'/result/'):
        os.mkdir('data/'+city+'/result/')


def isFriends(friends,a, b):

    friends_a=friends[friends.u1==a].u2
    return True if b in friends_a.values else False


def pickPairs(friends, i, SP, MAX_PAIRS,ulist):

    '''picks friend and stranger pairs
    Args:
        friends: friends list (asymetric) [u1, u2]
        i: iteration
        SP: list of stranger pairs
        MAX_PAIRS: number of existing friend pairs
        ulist: randomly shuffled user list
    Returns:
        pairs: [u1,u2,label]
    '''
    #print " permutation", i

    while len(ulist) >= 2:

        a = ulist.pop()
        b = ulist.pop()

        if not isFriends(friends, a,b):

            SP.append([a,b])
            if len(SP)>=MAX_PAIRS:
                return SP
        else:
            print ("friends found ", a,b)

    return SP


def make_allPairs(pairsFile, u_list_file, DATAPATH, friendFile, makeStrangers):

    '''gets friend and stranger pairs and writes to "clean_allPairs.csv"
    Args:
        friends: friends list (asymetric) [u1, u2] unordered pairs, duplicates exist
        u_list_file: dataset from which to read uids
    Returns:
        pairs: [u1,u2,label]
    '''

    u_list = pd.read_csv(DATAPATH + u_list_file).index.values

    friends = pd.read_csv(DATAPATH + friendFile)

    # take only pairs {u1, u2} where u1<u2, because {u2, u1} also exist but is a duplicate
    smallFriends=friends.loc[(friends.u1< friends.u2) & (friends.u1.isin(u_list))& (friends.u2.isin(u_list))].reset_index(drop=True)
    smallFriends["label"] = 1

    if makeStrangers:

        MAX_PAIRS, SP = len(smallFriends.u1), []
        #print MAX_PAIRS
        i = 0

        while len(SP) < MAX_PAIRS:

            SP = pickPairs(friends, i, SP, MAX_PAIRS, random.sample(u_list, k=len(u_list)))
            i += 1
        #print SP

        with open(DATAPATH + "strangers.csv", "wb") as f:

            for pair in SP:
                f.write(str(pair[0]) + ", " + str(pair[1]) + '\n')


    strangers=pd.read_csv(DATAPATH+"strangers.csv", names=['u1', 'u2'])
    strangers["label"]=0

    allPairs=smallFriends.append(strangers, ignore_index=True)
    #print "smallFriends.shape, strangers.shape", smallFriends.shape, strangers.shape, "allPairs.shape", allPairs.shape

    assert(len(allPairs)==len(smallFriends)*2 == len(strangers)*2)

    allPairs.to_csv(DATAPATH+ pairsFile)#, index=False)

    return allPairs


def pair_construct(u_list, friendFile, downSample):

    ''' construct users pairs
    Args:
        u_list: user list
        friends: file of DF of list of friends
        pairFile: store here for future
        downSample: Boolean True for word2vec features,
                    if False downsample later after calculation of overlap based features
    Returns:
        pair: u1, u2, label
    '''


    friends = pd.read_csv(DATAPATH + friendFile)

    # positive i.e. friend pairs
    pair_p = friends.loc[(friends.u1.isin(u_list)) & (friends.u2.isin(u_list))].copy()

    # sampling negative pairs , i.e. strangers
    pair_n = pd.DataFrame(pd.np.random.choice(u_list, 9 * pair_p.shape[0]), columns=['u1'])
    pair_n['u2'] = pd.np.random.choice(u_list, 9 * pair_p.shape[0])

    # remove dup user in pair
    pair_n = pair_n.loc[pair_n.u1 != pair_n.u2]
    # remove asymetric dups
    pair_n = pair_n.loc[pair_n.u1 < pair_n.u2]
    # remove dups
    pair_n = pair_n.drop_duplicates().reset_index(drop=True)

    # delete friends inside by setting the columns of the positive pairs to be indexes
    pair_n = pair_n.loc[~pair_n.set_index(list(pair_n.columns)).index.isin(pair_p.set_index(list(pair_p.columns)).index)]
    # now shuffle and reset the index
    pair_n = pair_n.loc[pd.np.random.permutation(pair_n.index)].reset_index(drop=True)

    if downSample:
        pair_n = pair_n.loc[0:1 * pair_p.shape[0] - 1, :]  # down sampling for emb features only

    pair_p['label'] = 1
    pair_n['label'] = 0
    print ("pair_n.shape, pair_p.shape", pair_n.shape, pair_p.shape)

    pair = pd.concat([pair_p, pair_n], ignore_index=True)
    pair = pair.reset_index(drop=True)

    return pair


