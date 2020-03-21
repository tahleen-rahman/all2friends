# Created by rahman at 17:01 2020-03-10 using PyCharm
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from gensim.models import word2vec
import multiprocessing as mp

from scipy.spatial import distance


def social_random_walk_core(DATAPATH, start_u, sym_pairs, walk_len, walk_times,  i, fold):

    ''' random walks from start_u on social network
    Args:
        model_name: dataset_method_parameter, e.g., enron_kDa_20
        start_u: starting user in a random walk
        sym_pairs: anonymized graph, symmetric
        walk_len: walk length
        walk_times: times start from each user
        i: iteration of cross val
        fold: fold of cross val
        Returns:
    '''

    np.random.seed()

    temp_walk = np.zeros((1, walk_len))# initialize random walk


    for w in range(walk_times):

        temp_walk[:, 0] = start_u#
        curr_u = start_u

        for j in range(walk_len-1):

            temp_val = sym_pairs.loc[sym_pairs.u1 == curr_u]
            next_u = np.random.choice(temp_val.u2.values, 1)[0]
            curr_u = next_u
            temp_walk[:, j+1] = next_u

        pd.DataFrame(temp_walk).to_csv(DATAPATH + "emb/" + i + "_" + fold + 'friends.walk', header=None, mode='a', index=False)


def para_friend_random_walk(DATAPATH, ulist,sym_pairs, walk_len, walk_times,core_num,  i, fold):

    ''' parallel random walk on incomplete user friend network
    Args:
        city: city
        model_name: 20_locid
        ulist: user list
        sym_pairs: edgeList
        walk_len: walk length
        walk_times: walk times
    Returns:
    '''

    # do not use shared memory
    Parallel(n_jobs = core_num)(delayed(social_random_walk_core)(DATAPATH, u, sym_pairs, walk_len, walk_times,  i, fold) for u in ulist)


def makeWalk(sym_pairs, DATAPATH, core_num, i, fold=""):
    ''' makes weighted bipartite graphs and calls the parallel random walk
    Args:
        city: city
        model_name: 20_locid
    Returns:
    '''

    walk_len, walk_times = 100, 20  # maximal 100 walk_len, 20 walk_times
    print ('walking, walk_len, walk_times, sym_pairs.shape, corenum,  i, fold', walk_len, walk_times, sym_pairs.shape, core_num, i, fold)
    para_friend_random_walk(DATAPATH, sym_pairs.u1.unique(), sym_pairs, walk_len, walk_times, core_num,  i, fold)



def emb_train(DATAPATH, i, fold, walk_len=100, walk_times=20, num_features=128):
    ''' train vector model
    Args:
        city: city
        model_name: 20_locid
        walk_len: walk length
        walk_times: walk times
        num_features: dimension for vector
    Returns:
    '''

    walks = pd.read_csv(DATAPATH  + "emb/" + i + "_" + fold + 'friends.walk', \
                        header=None, error_bad_lines=False)

    walks = walks.loc[np.random.permutation(len(walks))]
    walks = walks.reset_index(drop=True)
    walks = walks.applymap(str)  # gensim only accept list of strings

    print ('walk_len', walk_len, 'walk_times', walk_times, 'num_features', num_features)

    min_word_count = 10
    num_workers = mp.cpu_count()
    context = 10
    downsampling = 1e-3

    # gensim does not support numpy array, thus, walks.tolist()
    walks = walks.groupby(0).head(walk_times).values[:, :walk_len].tolist()

    emb = word2vec.Word2Vec(walks, workers=num_workers, \
                            size=num_features, min_count=min_word_count, \
                            window=context, sample=downsampling)

    print ('training done')
    emb.wv.save_word2vec_format(DATAPATH  + "emb/"  + i + "_" + fold + 'friends.emb')



def make_features_distances(DATAPATH, i):
    """
    pairwise distances as features, not shown in the paper
    :param DATAPATH:
    :return:
    """

    pair = pd.read_csv(DATAPATH + "HCI.csv", usecols=[0, 1, 2])

    for fold in range(0, 5):

        fold = str(fold)

        if os.path.exists(DATAPATH + i + "_" + fold + 'friendship.csv'):
            os.remove(DATAPATH + i + "_" + fold +'friendship.csv')

        emb = pd.read_csv(DATAPATH + "emb/" + i + "_" + fold +'friends.emb', header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0:'uid'})# last column is user id
        emb = emb.loc[emb.uid>0]# only take users, no loc_type, not necessary


        for i in range(len(pair)):

            u1 = pair.loc[i, 'u1']
            u2 = pair.loc[i, 'u2']

            label = pair.loc[i, 'label']

            u1_vector = emb.loc[emb.uid==u1, range(1, emb.shape[1])]
            u2_vector = emb.loc[emb.uid==u2, range(1, emb.shape[1])]
            print (u1.shape)
            try:
                i_feature = pd.DataFrame([[u1, u2, label, \
                                        distance.cosine(u1_vector, u2_vector), \
                                        distance.euclidean(u1_vector, u2_vector), \
                                        distance.correlation(u1_vector, u2_vector), \
                                        distance.chebyshev(u1_vector, u2_vector), \
                                        distance.braycurtis(u1_vector, u2_vector), \
                                        distance.canberra(u1_vector, u2_vector), \
                                        distance.cityblock(u1_vector, u2_vector), \
                                        distance.sqeuclidean(u1_vector, u2_vector)]])

                i_feature.to_csv(DATAPATH + i + "_" + fold + 'friendship.csv',  index = False, header = None, mode = 'a')
                #print "feature created"

            except Exception as e :
                print ("EXCEPTION!", u1_vector.shape, u2_vector.shape, u1, u2, e.message)

    return 'friendship.csv'





def friend2vec(DATAPATH, frn_train_file, i):
    """
    TODO
    create embeddings for the users in the pairs in HCI using word2vec on the incomplete network
    :param DATAPATH:
    :param frn_train_file: prefix of the file for each cross val iteration subgraph of friends in the training set to use for node2vec
    :return:
    """

    for fold in range(0,5):

        frn_train = pd.read_csv(DATAPATH +  i + "_" + str(fold) + frn_train_file) #, usecols=[0, 1, 2])

        # create symmetric pairs for random walk
        frn_train2 = pd.DataFrame()
        frn_train2['u1'] = frn_train.u2
        frn_train2['u2'] = frn_train.u1
        sym_pairs = frn_train2.append(frn_train.loc[:, ['u1', 'u2']])

        #sym_pairs.to_csv(DATAPATH + str(i) + 'sym_pairs.csv', index=False)
        #sym_pairs = pd.read_csv(DATAPATH + str(i) +  'sym_pairs.csv')

        if not os.path.exists(DATAPATH + 'emb/'):
            os.mkdir(DATAPATH + 'emb/')

        core_num = mp.cpu_count() - 1

        makeWalk(sym_pairs, DATAPATH, core_num, i, str(fold))

        emb_train(DATAPATH, i, str(fold))






def HADAMARD(u1, u2):
    return pd.np.multiply(u1, u2).values





def make_features_hada(DATAPATH, i=""):
    """
    calculates HADAMARD distance as features between the embedding pairs

    :param DATAPATH:
    :return:
    """

    pair = pd.read_csv(DATAPATH + "HCI.csv", usecols=[0, 1, 2])

    for fold in range(0, 5):

        fold = str(fold)

        if os.path.exists(DATAPATH + i + "_" + fold + 'friendship_hada.csv'):
            os.remove(DATAPATH + i + "_" + fold + 'friendship_hada.csv')

        emb = pd.read_csv(DATAPATH  + "emb/" + i + "_" + fold + 'friends.emb', header=None, skiprows=1, sep=' ')
        emb = emb.rename(columns={0: 'uid'})  # last column is user id

        count = 0

        for row in range(len(pair)):

            u1 = pair.loc[row, 'u1']
            u2 = pair.loc[row, 'u2']

            label = pair.loc[row, 'label']

            u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])]
            u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])]

            try:
                val = HADAMARD(u1_vector, u2_vector)

                i_feature = pd.DataFrame([[u1, u2, label]])

                for col in range(0, emb.shape[1]-1):
                    i_feature[col+3] = val[0][col]

                i_feature.to_csv(DATAPATH + i + "_" + fold + 'friendship_hada.csv',\
                                 index = False, header = None, mode = 'a')

            except Exception as e:
                print (sys.exc_info()[0])
                print (e)
                #print (u1, u2)
                count +=1

        print (count)

    return 'friendship_hada.csv'
