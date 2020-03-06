# Created by rahman at 14:51 2020-03-05 using PyCharm

import os
import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

classifiers={   'RF':(RandomForestClassifier, {"n_estimators": 101, "max_depth": 10}),
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
    while len(ulist)>= 2:
        a=ulist.pop()
        b=ulist.pop()
        if not isFriends(friends, a,b):
            SP.append([a,b])
            if len(SP)>=MAX_PAIRS:
                return SP
        else:
            print ("friends found ", a,b)

    return SP


def make_allPairs(pairsFile,  u_list_file, DATAPATH, friendFile, makeStrangers):

    '''gets friend and stranger pairs and writes to "clean_allPairs.csv"
    Args:
        friends: friends list (asymetric) [u1, u2]
        u_list_file: dataset from which to read uids
    Returns:
        pairs: [u1,u2,label]
    '''

    u_list = pd.read_csv(DATAPATH + u_list_file).index.values

    friends = pd.read_csv(DATAPATH + friendFile)

    smallFriends=friends.loc[(friends.u1< friends.u2) & (friends.u1.isin(u_list))& (friends.u2.isin(u_list))].reset_index(drop=True)
    smallFriends["label"]=1

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
