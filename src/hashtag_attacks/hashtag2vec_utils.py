# Created by rahman at 14:41 2020-03-09 using PyCharm
import pandas as pd
import traceback, os
from gensim.models import word2vec
from joblib import Parallel, delayed
import numpy as np
import multiprocessing as mp
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from shared_tools.utils import pair_construct


def uh_graph_build(ht):

    ''' build all the network
    Args:
        ht: hashtagsCleaned data
    Returns:
        uh_graph: graph data
        hu_graph:
    '''

    uh_graph = ht.groupby('uid')['ht_id'].value_counts()*1.0/ht.groupby('uid').size()
    uh_graph = pd.DataFrame(uh_graph)
    uh_graph.columns = ['weight']
    uh_graph = pd.DataFrame(uh_graph).reset_index()
    uh_graph.columns = ['uid', 'next', 'weight']

    hu_graph = ht.groupby('ht_id').uid.value_counts()*1.0/ht.groupby('ht_id').size()
    hu_graph = pd.DataFrame(hu_graph)
    hu_graph.columns = ['weight']
    hu_graph = pd.DataFrame(hu_graph).reset_index()
    hu_graph.columns = ['ht_id', 'next', 'weight']

    return uh_graph, hu_graph





def ul_random_walk_core( DATAPATH, model_name, start_u, uh_graph,hu_graph, walk_len, walk_times):

    ''' random walks from start_u on user location network
    Args:
        city: city
        model_name: 20_locid
        start_u: starting user in a random walk
        ul_graph, lu_graph: graph data (pandas df)
        walk_len: walk length
        walk_times: walk times
    Returns:
    '''

    np.random.seed()
    temp_walk = np.zeros((1, walk_len)) # initialize random walk

    for i in range(walk_times):

        temp_walk[:, 0] = start_u#
        curr_u = start_u
        flag = 0 # flag 0, user, flag 1, hashtag

        for j in range(walk_len-1):

            if flag == 0:# at user
                temp_val = uh_graph.loc[uh_graph.uid==curr_u]
                flag = 1
            elif flag == 1: # at hashtag
                temp_val = hu_graph.loc[hu_graph.ht_id==curr_u]
                flag = 0
            # sample with weights
            #print "flag, curr_u, temp_val['weight'].sum()",flag, curr_u, temp_val['weight'].sum()

            try:
                next_u = pd.np.random.choice(temp_val['next'].values, 1, p=temp_val['weight'])[0]
            except Exception as e:
                print ("EXCEPTION! e, temp_val['weight'].sum() , curr_u ", e, temp_val['weight'].sum(), curr_u)

            curr_u = next_u

            if flag == 1:
                temp_walk[:, j+1] = -next_u # location id is minus
            else:
                temp_walk[:, j+1] = next_u

        pd.DataFrame(temp_walk).to_csv(DATAPATH+'emb/' +model_name+'.walk', header=None, mode='a', index=False)



def para_ul_random_walk(DATAPATH, model_name, ulist, uh_graph,hu_graph, walk_len, walk_times,core_num):

    '''
    parallel random walk on user location network
    Args:
        city: city
        model_name: 20_locid
        ulist: user list
        uh_graph: edgeList
        walk_len: walk length
        walk_times: walk times
    Returns:
    '''

    # do not use shared memory
    Parallel(n_jobs = core_num)(delayed(ul_random_walk_core)( DATAPATH,model_name, u, uh_graph,hu_graph, walk_len, walk_times) for u in ulist)



def make_walk( DATAPATH,uh_graph, hu_graph, model_name, core_num):

    '''
    makes weighted bipartite graphs and calls the parallel random walk
    Args:
        city: city
        model_name: 20_locid
    Returns:
    '''

    ulist = uh_graph.uid.unique()
    walk_len, walk_times = 100, 20  # maximal 100 walk_len, 20 walk_times
    print ('walking, walk_len, walk_times, model_name, uh_graph.shape, hu_graph.columns', walk_len, walk_times, model_name, uh_graph.shape, hu_graph.columns)

    if not os.path.exists(DATAPATH + 'emb/'):
        os.mkdir(DATAPATH + 'emb/')

    para_ul_random_walk(DATAPATH, model_name, ulist, uh_graph, hu_graph, walk_len, walk_times, core_num)



def emb_train(DATAPATH,model_name, walk_len=100, walk_times=20, num_features=128):

    ''' train vector model
    Args:
        city: city
        model_name: 20_locid
        walk_len: walk length
        walk_times: walk times
        num_features: dimension for vector
    Returns:
    '''

    walks = pd.read_csv(DATAPATH+'emb/' +model_name+'.walk', \
                        header=None, error_bad_lines=False)

    walks = walks.loc[np.random.permutation(len(walks))]
    walks = walks.reset_index(drop=True)
    walks = walks.applymap(str)  # gensim only accept list of strings

    print ('walk_len', walk_len, 'walk_times', walk_times, 'num_features', num_features)

    min_word_count = 10
    num_workers = mp.cpu_count()-1
    context = 10
    downsampling = 1e-3

    # gensim does not support numpy array, thus, walks.tolist()
    walks = walks.groupby(0).head(walk_times).values[:, :walk_len].tolist()

    emb = word2vec.Word2Vec(walks,
                            workers=num_workers, \
                            size=num_features, min_count=min_word_count, \
                            window=context, sample=downsampling)

    print ('training done')

    emb.wv.save_word2vec_format(DATAPATH+'emb/'+ model_name + '_' + \
                                str(int(walk_len)) + '_' + str(int(walk_times)) + '_' + str(int(num_features)) + '.emb')





def feature_construct(DATAPATH, model_name, pairs, walk_len=100, walk_times=20, num_features=128):
    '''construct the feature matrixu2_checkin
    Args:
        city: city
        model_name: 20_locid
        pairs: friends n stranger list (asymetric) [u1, u2]
        walk_len: walk length
        walk_times: walk times
        num_features: dimension for vector
    Returns:
    '''

    data_file =  'emb/' + model_name + '_' + str(int(walk_len)) + '_' + str(int(walk_times)) + '_' + str(int(num_features)) + '.feature'

    if os.path.exists(DATAPATH + data_file):
        os.remove(DATAPATH + data_file)

    emb = pd.read_csv(DATAPATH+'emb/'+ model_name + '_' + \
                      str(int(walk_len)) + '_' + str(int(walk_times)) + '_' + str(int(num_features)) + '.emb', \
                      header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # last column is user id
    emb = emb.loc[emb.uid > 0]  # only take users, no loc_type, not necessary

    pair = pairs

    count=0
    for i in range(len(pair)):
        u1 = pair.loc[i, 'u1']
        u2 = pair.loc[i, 'u2']
        label = pair.loc[i, 'label']

        u1_vector = emb.loc[emb.uid == u1, range(1, emb.shape[1])].values
        u2_vector = emb.loc[emb.uid == u2, range(1, emb.shape[1])].values
        #print u1_vector.shape, u2_vector.shape
        try:
            i_feature = pd.DataFrame([[u1, u2, label, \
                                       cosine(u1_vector, u2_vector), \
                                       euclidean(u1_vector, u2_vector), \
                                       correlation(u1_vector, u2_vector), \
                                       chebyshev(u1_vector, u2_vector), \
                                       braycurtis(u1_vector, u2_vector), \
                                       canberra(u1_vector, u2_vector), \
                                       cityblock(u1_vector, u2_vector), \
                                       sqeuclidean(u1_vector, u2_vector)]])

            i_feature.to_csv(DATAPATH + data_file, index=False, header=None, mode='a')

        except ValueError:
            print (u1_vector.shape, u2_vector.shape)
            count+=1
    print  (count , " pairs not found out of ",  len(pair))

    return data_file





def predict_all_aucs( DATAPATH,model_name, walk_len=100, walk_times=20, num_features=128):

    ''' unsupervised prediction
    Args:
        city: city
        model_name: 20_locid
        walk_len: walk length
        walk_times: walk times
        num_features: dimension for vector
    Returns:
    '''
    dataset = pd.read_csv(DATAPATH+'emb/'+ model_name + '_' + \
                         str(int(walk_len))+'_'+str(int(walk_times))+'_'+str(int(num_features))+'.feature',\
                          names = ['u1', 'u2', 'label',\
                                   'cosine', 'euclidean', 'correlation', 'chebyshev',\
                                   'braycurtis', 'canberra', 'cityblock', 'sqeuclidean'])

    auc_res = []

    for i in ['cosine', 'euclidean', 'correlation', 'chebyshev',\
              'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']:

        i_auc = roc_auc_score(dataset.label, dataset[i])
        if i_auc < 0.5: i_auc = 1-i_auc
        print (i, i_auc)
        auc_res.append(i_auc)

    pd.DataFrame([auc_res], columns=['cosine', 'euclidean', 'correlation', 'chebyshev',\
                            'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']).to_csv(\
                            DATAPATH + 'emb/' + model_name + '_' + \
                            str(int(walk_len))+'_'+str(int(walk_times))+'_'+str(int(num_features))+'.result', index=False)



    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    train_pairs = pd.read_csv(DATAPATH + 'train_pairs.csv', index_col=0)
    test_pairs = pd.read_csv(DATAPATH + 'test_pairs.csv', index_col=0)

    train_set = dataset.merge(train_pairs).dropna()
    test_set = dataset.merge(test_pairs).dropna()

    print ("train_set.shape, test_set.shape", train_set.shape, test_set.shape)
    print (" dataset[dataset.label==0].shape,  dataset[dataset.label==1].shape", dataset[dataset.label == 0].shape,  dataset[dataset.label == 1].shape)
    print ("train_set[train_set.label==0].shape, train_set[train_set.label==1].shape", train_set[train_set.label == 0].shape, train_set[train_set.label == 1].shape)
    print ("test_set[test_set.label == 0].shape, test_set[test_set.label == 1].shape", test_set[test_set.label == 0].shape, test_set[test_set.label == 1].shape)

    X_train, y_train = train_set.iloc[:, 3:].values, train_set.iloc[:, 2].values
    X_test, y_test = test_set.iloc[:, 3:].values, test_set.iloc[:, 2].values

    classifier = clf.fit(X_train, y_train)
    pred_proba = clf.predict_proba(X_test)

    print ("roc_auc_score(y_test, pred_proba[:,1])", roc_auc_score(y_test, pred_proba[:, 1]))




def sup_feature_construct( DATAPATH,metrics, name,  model_name,  dataFile, friendFile, walk_len=100, walk_times=20, num_features=128):
    '''
    construct the features for each pair according to distance metric from the word2vec embedding of each user
    and writes to file
    Args:
        metrics: dictionary of distance metrics ('L2': L2,  'AVG':AVG, 'L1':L1,'HADAMARD': HADAMARD, 'concat':concat )
        name: choice of the similarity metric that instantiates a function
        friends: Active friends DF (symmetric) [u1, u2]
        city: city
        model_name: 20_locid
        walk_len: walk length
        walk_times: walk times
        num_features: dimension for vector
    Returns:
    '''

    emb = pd.read_csv(DATAPATH+'emb/'+ model_name + '_' + \
                      str(int(walk_len)) + '_' + str(int(walk_times)) + '_' + str(int(num_features)) + '.emb', \
                      header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # first column is user id
    emb = emb.loc[emb.uid > 0]  # only take users negatives are for hahstags

    friends = pd.read_csv(DATAPATH+friendFile)
    allPairs = pair_construct(emb.uid.unique(), friends, downSample=True)


    emb.set_index('uid', inplace=True)
    emb.index = emb.index.astype(pd.np.int64)


    dataset = allPairs
    print ("making dataset file ", dataFile, " from ", len(dataset), " pairs")

    dataset['cosine_Dist']=-99
    for colName in emb.columns:
        dataset[colName]=-99
    print (len(emb.columns), "columns set to -99")
    #print dataset.columns

    count = 0
    print ("making dataset acc to metric", name)

    for row in dataset.index:
        try:
            dataset.loc[row, 'cosine_Dist'] = cosine(emb.loc[dataset.loc[row, 'u1']], emb.loc[dataset.loc[row, 'u2']])
            diff=metrics[name](emb.loc[dataset.loc[row, 'u1']] ,emb.loc[dataset.loc[row, 'u2']])
            dataset.iloc[row, -len(emb.columns):]= diff
        except Exception as e:
            count+=1

    dataset.dropna(inplace=True)
    dataset.to_csv(dataFile, index=False)

    #print "dataset.columns", dataset.columns
    frns = dataset[dataset.label == 1]
    strangers = dataset[dataset.label == 0]

    print ("dataset.shape ",dataset.shape, "no of pairs not found in avgs ", count," len(frns), len(strangers)",  len(frns), len(strangers))

    return dataset

