# Created by rahman at 16:54 2020-03-10 using PyCharm
import os

import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def split_train_test(DATAPATH):
    """

    :param DATAPATH:
    :return:
    """
    if os.path.exists(DATAPATH + "HCI.csv"):
        pairs = pd.read_csv(DATAPATH + "HCI.csv", usecols=[0, 1, 2])

    else:
        print("pairs HCI.csv not found, Creating it using makeHCI")
        from src.multimodal_ensemble.multimodal_utils import makeHCI
        makeHCI(DATAPATH)

    str_train = pairs[pairs.label == 0].sample(frac=0.8)
    str_test = pairs[pairs.label == 0].drop(str_train.index)
    frn_train = pairs[pairs.label == 1].sample(frac=0.8)
    frn_test = pairs[pairs.label == 1].drop(frn_train.index)

    train_pairs = frn_train.append(str_train)
    train_pairs.to_csv(DATAPATH + 'train_pairs.csv')

    test_pairs=frn_test.append(str_test)
    test_pairs.to_csv(DATAPATH + 'test_pairs.csv')
    return frn_train


def makeHCI(DATAPATH):
    """
    outer join of all pairs that were formed by all 3 modalities
    :return:
    HCI : union of all pairs
    cap_dataset, ht_dataset, im_dataset
    """

    im_dataset  = pd.read_csv(DATAPATH + "im_dataset.csv")

    ht_dataset  = pd.read_csv(DATAPATH + 'ht_dataset.csv')

    cap_dataset = pd.read_csv(DATAPATH + 'cap_dataset.csv')

    print(im_dataset.shape)
    print(ht_dataset.shape)

    IH = im_dataset[['u1', 'u2', 'label']].merge(ht_dataset[['u1', 'u2', 'label']], how='outer', on=['u1', 'u2', 'label'])

    print(IH.columns)

    HCI = cap_dataset[['u1', 'u2', 'label']].merge(IH, how='outer', on=['u1', 'u2', 'label'])

    print("HCI.shape", HCI.shape)

    print("HCI.dropna().shape ", HCI.dropna().shape)
    print("HCI.dropna(how='all').shape ", HCI.dropna(how='all').shape)

    print(HCI.columns)

    HCI.to_csv(DATAPATH + 'HCI.csv', index=False)

    return HCI, cap_dataset, ht_dataset, im_dataset


def recalculate_missingHCI(DATAPATH, HCI, cap_dataset, ht_dataset, im_dataset):
    """
    try to recalculate missing data for pairs that we have missed from one of the modalities but could have calculated.
    Can happen for stranger pairs, should not have happened for friend pairs

    :param DATAPATH:
    :param HCI: outer join of all pairs that were formed by 3 modalities
    :param cap_dataset:
    :param ht_dataset:
    :param im_dataset:
    :return:
    """

    cap_list, ht_list, im_list = [], [],[]

    # load image dataset dependencies
    catcounts = pd.read_csv(DATAPATH + "im_dataset.csv", index_col=0) #"100_imgFilt_prob05c_avg_01c_catFilt_100_dataset.csv"
    im_uid = catcounts.index.tolist()
    ent_df = pd.read_csv(DATAPATH  + "entropy_of_categories.csv", index_col=0)


    #avgs = pd.read_csv(DATAPATH + "100_imgFilt_prob05c_avg_01c_catFilt_100.csv", index_col=0)
    #im_uid = avgs.index.tolist()


    # load caption dataset dependences
    df = pd.read_csv(DATAPATH+ "words.csv")
    cap_uid = df.uid.unique().tolist()
    TFIDF_matrix = scipy.sparse.load_npz(DATAPATH + 'True0.01filt_TFIDF.npz')



    # load ht dependencies
    ht = pd.read_csv(DATAPATH + 'ht_dataset.csv')
    ht_uid = ht.uid.unique()

    gp = ht.groupby('uid')['ht_id']
    htids = {}
    for userid, val in gp:
        htids[userid] = val.values
    cnt_df = pd.read_csv(DATAPATH + "counts_entropies.csv")
    countdf = ht.groupby('ht_id')['uid'].count().reset_index(name='counts')

    from scipy.spatial import distance

    for i in HCI.index:

        u1, u2, label = HCI.loc[i, 'u1'], HCI.loc[i, 'u2'], HCI.loc[i, 'label']

        #### HASHTAGS ####
        ht_row = ht_dataset[(ht_dataset.u1 == u1) & (ht_dataset.u2 == u2)].dropna()#pd.isnull(HCI.loc[i, 'count_aa_ent']):

        # if a pair  in HCI does not exist in ht_dataset, try to calculate it
        if ht_row.shape[0] < 1:

            if u1 in ht_uid and u2 in ht_uid:

                overlap = list(set(htids[u1]).intersection(set(htids[u2])))
                if len(overlap) > 0:
                    v2 = gp.get_group(u2).value_counts()[overlap]
                    v1 = gp.get_group(u1).value_counts()[overlap]
                    aa = pd.np.sum(pd.np.reciprocal(cnt_df[cnt_df.ht_id.isin(overlap)].entropy))
                    unionn = set(htids[u1]).union(set(htids[u2]))

                    ht_list.append([u1, u2, label, \
                                    distance.cosine(v1, v2),   \
                                    pd.np.dot(v1, v2), \
                                    len(overlap), \
                                    len(overlap) * 1.0 / len(unionn), \
                                    cnt_df[cnt_df.ht_id.isin(overlap)].entropy.min(), \
                                     aa, \
                                    countdf[countdf.ht_id.isin(overlap)].counts.min(),  \
                                    pd.np.sum(pd.np.reciprocal(pd.np.log(countdf[countdf.ht_id.isin(overlap)].counts))),  \
                                    aa * 1.0 / len(unionn), \
                                    aa * 1.0 / pd.np.sum(pd.np.reciprocal(cnt_df[cnt_df.ht_id.isin(unionn)].entropy))]) \

                    print('ht')

        elif len(ht_row) > 1:

            print("duplicate ht rows found", ht_row)


        #### CAPTIONS ####
        cap_row = cap_dataset[(cap_dataset.u1 == u1) & (cap_dataset.u2 == u2)].dropna()

        # if a pair  in HCI does not exist in cap_dataset, try to calculate it
        if cap_row.shape[0]<1:

            if u1 in cap_uid and u2 in cap_uid:

                pos1i, pos2i = df[df.uid == u1].index, df[df.uid == u2].index
                pos1arr, pos2arr = pos1i.get_values(), pos2i.get_values()
                pos1, pos2 = pos1arr[0], pos2arr[0]

                u1_vector, u2_vector = TFIDF_matrix[pos1, :].toarray(), TFIDF_matrix[pos2, :].toarray()
                cap_list.append([u1, u2, label, \
                                           distance.cosine(u1_vector, u2_vector), \
                                           distance.euclidean(u1_vector, u2_vector), \
                                           distance.correlation(u1_vector, u2_vector), \
                                           distance.chebyshev(u1_vector, u2_vector), \
                                           distance.braycurtis(u1_vector, u2_vector), \
                                           distance.canberra(u1_vector, u2_vector), \
                                           distance.cityblock(u1_vector, u2_vector), \
                                           distance.sqeuclidean(u1_vector, u2_vector)])
                print('cap')

        elif len(cap_row)>1:

            print("duplicate cap rows found", cap_row)


        #### IMAGES ####
        im_row = im_dataset[(im_dataset.u1 == u1) & (im_dataset.u2 == u2)].dropna()

        # if a pair  in HCI does not exist in im_dataset, try to calculate it
        if im_row.shape[0]<1:

            if u1 in im_uid and u2 in im_uid:

                u1_vector, u2_vector=catcounts.loc[u1], catcounts.loc[u2]
                mini = pd.np.minimum(u1_vector, u2_vector)

                lst=[u1, u2, label]
                lst.extend(mini.values)
                lst.extend(distance.cosine(u1_vector, u2_vector))
                lst.extend([pd.np.max(mini.values), ent_df.loc[int(pd.np.argmax(mini.values))].entropy])
                im_list.append(lst)

                print('im')

        elif len(im_row)>1:

            print("duplicate im rows found", im_row)




    ### append the newly calculated  feats  to each modality###

    extra_cap_dataset = cap_dataset.append(pd.DataFrame(cap_list, \
                                                        columns=['u1', 'u2', 'label', \
                                                                  'cosine', 'euclidean', 'correlation', 'chebyshev',
                                                                   'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']),\
                                                        ignore_index=True)

    print(cap_dataset.shape, extra_cap_dataset.dropna().shape)




    imcols = ['u1', 'u2', 'label']
    for i in range(0, len(catcounts.columns)):
         imcols.append("comp_" + str(i))
    imcols.extend(['cosine_Dist','max_count','ent_maxcat'])

    extra_im_dataset= im_dataset.append(pd.DataFrame(im_list, columns=imcols), ignore_index=True)

    print(im_dataset.shape, extra_im_dataset.dropna().shape)




    extra_ht_dataset= ht_dataset.append(pd.DataFrame(ht_list, columns=['u1', 'u2', 'label', \
                                                                                 'cos_dis', 'dot_pro', 'common_h', 'overlap_h', \
                                                                                 'min_ent', 'aa_ent', 'min_h','aa_h', 'count_aa_ent',\
                                                                                 'big_aa_ent']), ignore_index=True)



    print(extra_ht_dataset.shape, extra_ht_dataset.dropna().shape)



    extra_cap_dataset.to_csv(DATAPATH + 'extra_cap_dataset.csv', index=False)
    extra_ht_dataset.to_csv(DATAPATH + 'extra_ht_dataset.csv', index=False)
    extra_im_dataset.to_csv(DATAPATH + 'extra_im_dataset.csv', index=False)

    return 'extra_cap_dataset.csv', 'extra_ht_dataset.csv', 'extra_im_dataset.csv'


def write_posteriors(cap_file, ht_file, im_file, loc_file, network_file, DATAPATH):
    """
    train 5 classifiers for each of the 5 modalities using all train set pairs  
    and write the posterior probabilities for the test pairs
    :param cap_file: 
    :param ht_file: 
    :param im_file: 
    :param loc_file: 
    :param network_file: 
    :param DATAPATH: 
    :return: 
    """

    ht_dataset = pd.read_csv(DATAPATH + ht_file)

    im_dataset = pd.read_csv(DATAPATH + im_file)

    #im_dataset.drop('maxcat', inplace=True, axis=1)

    cap_dataset = pd.read_csv(DATAPATH + cap_file)

    loc_dataset = pd.read_csv(DATAPATH + loc_file)

    cols = ['u1', 'u2', 'label']
    cols.extend(range(1, 129))
    friendship_dataset = pd.read_csv(DATAPATH + network_file, names = cols)


    train_pairs = pd.read_csv(DATAPATH + 'train_pairs.csv', index_col=0)
    test_pairs = pd.read_csv(DATAPATH + 'test_pairs.csv', index_col=0)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    count = 0

    for dataset in [cap_dataset, im_dataset, ht_dataset, loc_dataset, friendship_dataset]:

        train_set = dataset.merge(train_pairs).dropna()
        test_set = dataset.merge(test_pairs).dropna()

        for col in train_set.columns:
            train_set[col] = pd.to_numeric(train_set[col])

        for col in test_set.columns:
            test_set[col] = pd.to_numeric(test_set[col])


        print("train_set.shape, test_set.shape", train_set.shape, test_set.shape)
        print(" dataset[dataset.label==0].shape,  dataset[dataset.label==1].shape", dataset[dataset.label==0].shape,  dataset[dataset.label==1].shape)
        print("train_set[train_set.label==0].shape, train_set[train_set.label==1].shape", train_set[train_set.label==0].shape, train_set[train_set.label==1].shape)
        print("test_set[test_set.label == 0].shape, test_set[test_set.label == 1].shape", test_set[test_set.label == 0].shape, test_set[test_set.label == 1].shape)

        X_train, y_train = train_set.iloc[:, 3:].values, train_set.iloc[:, 2].values
        X_test, y_test = test_set.iloc[:, 3:].values, test_set.iloc[:, 2].values

        classifier = clf.fit(X_train, y_train)

        pred_proba = clf.predict_proba(X_test)

        pred_proba_train = clf.predict_proba(X_train)
        print("roc_auc_score(y_train, pred_proba_train[:, 1])", roc_auc_score(y_train, pred_proba_train[:, 1]))

        print("roc_auc_score(y_test, pred_proba[:,1])", roc_auc_score(y_test, pred_proba[:, 1]))

        test_set['predproba_1'] = pred_proba[:, 1]

        test_set.to_csv(DATAPATH + str(count) + 'test_set.csv')  # 5793
        
        count += 1


def unite_posteriors(DATAPATH):
    """
    combine the posteriors from all 5 test set files created by write_posteriors
    :param DATAPATH: 
    :return: 
    """
    

    cap_testset = pd.read_csv(DATAPATH + '0test_set.csv', index_col=0)
    im_testset = pd.read_csv(DATAPATH + '1test_set.csv', index_col=0)
    ht_testset = pd.read_csv(DATAPATH + '2test_set.csv', index_col=0)
    loc_testset = pd.read_csv(DATAPATH + '3test_set.csv', index_col=0)
    frn_testset = pd.read_csv(DATAPATH + '4test_set.csv', index_col=0)

    cap_count, ht_count, loc_count, im_count, frn_count = 0, 0, 0, 0, 0

    test_pairs = pd.read_csv(DATAPATH + 'test_pairs.csv', index_col=0)

    for i in test_pairs.index:
        #print(i
        u1, u2 = test_pairs.loc[i, 'u1'], test_pairs.loc[i, 'u2']

        cap_row = cap_testset[(cap_testset.u1 == u1) & (cap_testset.u2 == u2)].dropna()
        if cap_row.shape[0] ==1 :
            test_pairs.loc[i, 'cap_prob_1'] = cap_row.predproba_1.values
        else:
            cap_count += 1

        ht_row = ht_testset[(ht_testset.u1 == u1) & (ht_testset.u2 == u2)].dropna()
        if ht_row.shape[0] == 1:
            test_pairs.loc[i, 'ht_prob_1'] = ht_row.predproba_1.values
        else:
            ht_count += 1

        loc_row = loc_testset[(loc_testset.u1 == u1) & (loc_testset.u2 == u2)].dropna()
        if loc_row.shape[0] == 1:
            test_pairs.loc[i, 'loc_prob_1'] = loc_row.predproba_1.values
        else:
            loc_count += 1

        im_row = im_testset[(im_testset.u1 == u1) & (im_testset.u2 == u2)].dropna()
        if im_row.shape[0] == 1:
            test_pairs.loc[i, 'im_prob_1'] = im_row.predproba_1.values
        else:
            im_count += 1

        frn_row = frn_testset[(frn_testset.u1 == u1) & (frn_testset.u2 == u2)].dropna()
        if frn_row.shape[0] == 1:
            test_pairs.loc[i, 'frn_prob_1'] = frn_row.predproba_1.values
        else:
            frn_count += 1

    print("cap_count, ht_count, loc_count, im_count, frn_count")
    print("not found or duplicates", cap_count, ht_count, loc_count, im_count, frn_count)
    print("test_pairs.shape", test_pairs.shape)
    print("test_pairs.dropna().shape", test_pairs.dropna().shape)

    test_pairs.to_csv(DATAPATH + 'all_probs_1.csv')
    
    
    
    
def score_avg5probs(DATAPATH):
    """
    baseline multimodal adversary that simply averages
    :param DATAPATH: 
    :return: 
    """

    test_pairs = pd.read_csv(DATAPATH + 'all_probs_1.csv', index_col=0)
    print(test_pairs.shape)
    test_pairs['avg_prob_1'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'loc_prob_1', 'frn_prob_1', 'im_prob_1']].mean(axis=1)
    drop_ind = test_pairs[test_pairs.avg_prob_1.isna()].index
    test_pairs.drop(drop_ind, inplace=True)
    print(test_pairs.shape)

    print(roc_auc_score(test_pairs.label, test_pairs.avg_prob_1))
    test_pairs.to_csv(DATAPATH + 'avg_probs_1.csv', index=False)
    
    
    




def score_subsets_weighted(DATAPATH):
    """
    our final multimodal adversary 
    :param DATAPATH: 
    :return: 
    """
    cap_testset = pd.read_csv(DATAPATH + '0test_set.csv', index_col=0)
    im_testset = pd.read_csv(DATAPATH + '1test_set.csv', index_col=0)
    ht_testset = pd.read_csv(DATAPATH + '2test_set.csv', index_col=0)
    loc_testset = pd.read_csv(DATAPATH + '3test_set.csv', index_col=0)
    frn_testset = pd.read_csv(DATAPATH + '4test_set.csv', index_col=0)

    cap_roc = roc_auc_score(cap_testset.label, cap_testset.predproba_1)
    im_roc = roc_auc_score(im_testset.label, im_testset.predproba_1)
    ht_roc = roc_auc_score(ht_testset.label, ht_testset.predproba_1)
    loc_roc = roc_auc_score(loc_testset.label, loc_testset.predproba_1)
    frn_roc = roc_auc_score(frn_testset.label, frn_testset.predproba_1)

    weighted_test_pairs = pd.read_csv(DATAPATH + 'all_probs_1.csv', index_col=0)

    weighted_test_pairs['cap_prob_1'] = weighted_test_pairs.cap_prob_1 * cap_roc
    weighted_test_pairs['im_prob_1'] = weighted_test_pairs.im_prob_1 * im_roc
    weighted_test_pairs['ht_prob_1'] = weighted_test_pairs.ht_prob_1 * ht_roc
    weighted_test_pairs['loc_prob_1'] = weighted_test_pairs.loc_prob_1 * loc_roc
    weighted_test_pairs['frn_prob_1'] = weighted_test_pairs.frn_prob_1 * frn_roc
    print(cap_roc, im_roc, ht_roc, loc_roc, frn_roc)

    test_pairs = weighted_test_pairs

    print(test_pairs.shape)
    test_pairs['avg_prob_1'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'loc_prob_1', 'frn_prob_1', 'im_prob_1']].mean(axis=1)
    drop_ind = test_pairs[test_pairs.avg_prob_1.isna()].index
    test_pairs.drop(drop_ind, inplace=True)
    print(test_pairs.shape)

    print(round(roc_auc_score(test_pairs.label, test_pairs.avg_prob_1), 3))

    test_pairs['hlic'] = test_pairs[['im_prob_1', 'ht_prob_1', 'loc_prob_1', 'cap_prob_1']].mean(axis=1)
    test_pairs_hlic= test_pairs[['u1', 'u2', 'label', 'hlic']].dropna()
    print("roc_auc_score(test_pairs_hlic.label, test_pairs_hlic.hlic), test_pairs_hlic.shape", \
        round(roc_auc_score(test_pairs_hlic.label, test_pairs_hlic.hlic), 3), test_pairs_hlic.shape)

    test_pairs['hlie'] = test_pairs[['im_prob_1', 'ht_prob_1', 'loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_hlie= test_pairs[['u1', 'u2', 'label', 'hlie']].dropna()
    print("roc_auc_score(test_pairs_hlie.label, test_pairs_hlie.hlie), test_pairs_hlie.shape", \
        round(roc_auc_score(test_pairs_hlie.label, test_pairs_hlie.hlie), 3), test_pairs_hlie.shape)

    test_pairs['chle'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_chle = test_pairs[['u1', 'u2', 'label', 'chle']].dropna()
    print("roc_auc_score(test_pairs_chle.label, test_pairs_chle.chle), test_pairs_chle.shape", \
        round(roc_auc_score(test_pairs_chle.label, test_pairs_chle.chle), 3), test_pairs_chle.shape)

    test_pairs['clie'] = test_pairs[['cap_prob_1', 'loc_prob_1', 'im_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_clie = test_pairs[['u1', 'u2', 'label', 'clie']].dropna()
    print("roc_auc_score(test_pairs_clie.label, test_pairs_clie.clie), test_pairs_clie.shape", \
        round(roc_auc_score(test_pairs_clie.label, test_pairs_clie.clie),3 ), test_pairs_clie.shape)

    test_pairs['cle'] = test_pairs[['cap_prob_1', 'loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_cle = test_pairs[['u1', 'u2', 'label', 'cle']].dropna()
    print("roc_auc_score(test_pairs_cle.label, test_pairs_cle.cle), test_pairs_cle.shape", \
        round(roc_auc_score(test_pairs_cle.label, test_pairs_cle.cle), 3), test_pairs_cle.shape)

    test_pairs['ile'] = test_pairs[['im_prob_1', 'loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_ile = test_pairs[['u1', 'u2', 'label', 'ile']].dropna()
    print("roc_auc_score(test_pairs_ile.label, test_pairs_ile.ile), test_pairs_ile.shape", \
        round(roc_auc_score(test_pairs_ile.label, test_pairs_ile.ile),3 ), test_pairs_ile.shape)

    test_pairs['chie'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'im_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_chie = test_pairs[['u1', 'u2', 'label', 'chie']].dropna()
    print("roc_auc_score(test_pairs_chie.label, test_pairs_chie.chie), test_pairs_chie.shape", \
        round(roc_auc_score(test_pairs_chie.label, test_pairs_chie.chie),3 ), test_pairs_chie.shape)

    test_pairs['cie'] = test_pairs[['cap_prob_1', 'im_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_cie = test_pairs[['u1', 'u2', 'label', 'cie']].dropna()
    print("roc_auc_score(test_pairs_cie.label, test_pairs_cie.cie), test_pairs_cie.shape", \
        round(roc_auc_score(test_pairs_cie.label, test_pairs_cie.cie),3 ), test_pairs_cie.shape)

    test_pairs['hle'] = test_pairs[['ht_prob_1', 'loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_hle = test_pairs[['u1', 'u2', 'label', 'hle']].dropna()
    print("roc_auc_score(test_pairs_hle.label, test_pairs_hle.hle), test_pairs_hle.shape", \
        round(roc_auc_score(test_pairs_hle.label, test_pairs_hle.hle),3 ), test_pairs_hle.shape)

    test_pairs['hie'] = test_pairs[['ht_prob_1', 'im_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_hie = test_pairs[['u1', 'u2', 'label', 'hie']].dropna()
    print("roc_auc_score(test_pairs_hie.label, test_pairs_hie.hie), test_pairs_hie.shape", \
        round(roc_auc_score(test_pairs_hie.label, test_pairs_hie.hie),3 ), test_pairs_hie.shape)

    test_pairs['che'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_che = test_pairs[['u1', 'u2', 'label', 'che']].dropna()
    print("roc_auc_score(test_pairs_che.label, test_pairs_che.che), test_pairs_che.shape", \
        round(roc_auc_score(test_pairs_che.label, test_pairs_che.che),3 ), test_pairs_che.shape)

    test_pairs['eh'] = test_pairs[[ 'ht_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_eh = test_pairs[['u1', 'u2', 'label', 'eh']].dropna()
    print("roc_auc_score(test_pairs_eh.label, test_pairs_eh.eh), test_pairs_eh.shape", \
        round(roc_auc_score(test_pairs_eh.label, test_pairs_eh.eh),3 ), test_pairs_eh.shape)

    test_pairs['ce'] = test_pairs[['cap_prob_1',  'frn_prob_1']].mean(axis=1)
    test_pairs_ce = test_pairs[['u1', 'u2', 'label', 'ce']].dropna()
    print("roc_auc_score(test_pairs_ce.label, test_pairs_ce.ce), test_pairs_ce.shape", \
        round(roc_auc_score(test_pairs_ce.label, test_pairs_ce.ce),3 ), test_pairs_ce.shape)

    test_pairs['le'] = test_pairs[['loc_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_le = test_pairs[['u1', 'u2', 'label', 'le']].dropna()
    print("roc_auc_score(test_pairs_le.label, test_pairs_le.le), test_pairs_le.shape", \
        round(roc_auc_score(test_pairs_le.label, test_pairs_le.le),3 ), test_pairs_le.shape)

    test_pairs['ie'] = test_pairs[['im_prob_1', 'frn_prob_1']].mean(axis=1)
    test_pairs_ie = test_pairs[['u1', 'u2', 'label', 'ie']].dropna()
    print("roc_auc_score(test_pairs_ie.label, test_pairs_ie.ie), test_pairs_ie.shape", \
        round(roc_auc_score(test_pairs_ie.label, test_pairs_ie.ie),3 ), test_pairs_ie.shape)


    test_pairs['hli'] = test_pairs[['im_prob_1', 'ht_prob_1', 'loc_prob_1']].mean(axis=1)
    test_pairs_hli = test_pairs[['u1', 'u2', 'label', 'hli']].dropna()
    print("roc_auc_score(test_pairs_hli.label, test_pairs_hli.hli), test_pairs_hli.shape", \
        roc_auc_score(test_pairs_hli.label, test_pairs_hli.hli), test_pairs_hli.shape)

    test_pairs['chl'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'loc_prob_1']].mean(axis=1)
    test_pairs_chl = test_pairs[['u1', 'u2', 'label', 'chl']].dropna()
    print("roc_auc_score(test_pairs_chl.label, test_pairs_chl.chl), test_pairs_chl.shape", \
        roc_auc_score(test_pairs_chl.label, test_pairs_chl.chl), test_pairs_chl.shape)

    test_pairs['cli'] = test_pairs[['cap_prob_1', 'loc_prob_1', 'im_prob_1']].mean(axis=1)
    test_pairs_cli = test_pairs[['u1', 'u2', 'label', 'cli']].dropna()
    print("roc_auc_score(test_pairs_cli.label, test_pairs_cli.cli), test_pairs_cli.shape", \
        roc_auc_score(test_pairs_cli.label, test_pairs_cli.cli), test_pairs_cli.shape)

    test_pairs['cl'] = test_pairs[['cap_prob_1', 'loc_prob_1']].mean(axis=1)
    test_pairs_cl = test_pairs[['u1', 'u2', 'label', 'cl']].dropna()
    print("roc_auc_score(test_pairs_cl.label, test_pairs_cl.cl), test_pairs_cl.shape", \
        roc_auc_score(test_pairs_cl.label, test_pairs_cl.cl), test_pairs_cl.shape)

    test_pairs['il'] = test_pairs[['im_prob_1', 'loc_prob_1']].mean(axis=1)
    test_pairs_il = test_pairs[['u1', 'u2', 'label', 'il']].dropna()
    print("roc_auc_score(test_pairs_il.label, test_pairs_il.il), test_pairs_il.shape", \
        roc_auc_score(test_pairs_il.label, test_pairs_il.il), test_pairs_il.shape)

    test_pairs['chi'] = test_pairs[['cap_prob_1', 'ht_prob_1', 'im_prob_1']].mean(axis=1)
    test_pairs_chi = test_pairs[['u1', 'u2', 'label', 'chi']].dropna()
    print("roc_auc_score(test_pairs_chi.label, test_pairs_chi.chi), test_pairs_chi.shape", \
        roc_auc_score(test_pairs_chi.label, test_pairs_chi.chi), test_pairs_chi.shape)

    test_pairs['ci'] = test_pairs[['cap_prob_1', 'im_prob_1']].mean(axis=1)
    test_pairs_ci = test_pairs[['u1', 'u2', 'label', 'ci']].dropna()
    print("roc_auc_score(test_pairs_ci.label, test_pairs_ci.ci), test_pairs_ci.shape", \
        roc_auc_score(test_pairs_ci.label, test_pairs_ci.ci), test_pairs_ci.shape)

    test_pairs['hl'] = test_pairs[['ht_prob_1', 'loc_prob_1']].mean(axis=1)
    test_pairs_hl = test_pairs[['u1', 'u2', 'label', 'hl']].dropna()
    print("roc_auc_score(test_pairs_hl.label, test_pairs_hl.hl), test_pairs_hl.shape", \
        roc_auc_score(test_pairs_hl.label, test_pairs_hl.hl), test_pairs_hl.shape)

    test_pairs['hi'] = test_pairs[['ht_prob_1', 'im_prob_1']].mean(axis=1)
    test_pairs_hi = test_pairs[['u1', 'u2', 'label', 'hi']].dropna()
    print("roc_auc_score(test_pairs_hi.label, test_pairs_hi.hi), test_pairs_hi.shape", \
        roc_auc_score(test_pairs_hi.label, test_pairs_hi.hi), test_pairs_hi.shape)

    test_pairs['ch'] = test_pairs[['cap_prob_1', 'ht_prob_1']].mean(axis=1)
    test_pairs_ch = test_pairs[['u1', 'u2', 'label', 'ch']].dropna()
    print("roc_auc_score(test_pairs_ch.label, test_pairs_ch.ch), test_pairs_ch.shape", \
        roc_auc_score(test_pairs_ch.label, test_pairs_ch.ch), test_pairs_ch.shape)

    test_pairs.to_csv(DATAPATH + 'weighted_avg_probs.csv', index=False)
