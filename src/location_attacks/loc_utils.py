# Created by rahman at 12:00 2020-03-10 using PyCharm
import os

import pandas as pd
from scipy.spatial import distance


def make_features(city, DATAPATH, data_file):

    """
    calculate features i.e. pairwise distances between location embeddings per user for each user pair if embeddings exist
    :param city:
    :param DATAPATH:
    :return:
    """
    if os.path.exists(DATAPATH + "HCI.csv"):
        pair = pd.read_csv(DATAPATH + "HCI.csv", usecols=[0, 1, 2])

    else:
        print("pairs HCI.csv not found, Creating it using makeHCI")
        from src.multimodal_ensemble.multimodal_utils import makeHCI
        makeHCI(DATAPATH)


    print ("pair.columns", pair.columns)

    emb = pd.read_csv(DATAPATH + city + '_locid_100_20_128.emb', header=None, skiprows=1, sep=' ')
    emb = emb.rename(columns={0: 'uid'})  # last column is user id
    emb = emb.loc[emb.uid > 0]  # in the .emb file, users embeddings start with positive values and location embeddings start with negative values

    emb.to_csv(DATAPATH + "locBYuser.csv", index=False)

    #emb = pd.read_csv(DATAPATH + 'locBYuser.csv', index_col=0)
    #emb_uid = emb.index.tolist()

    emb_uid = emb.uid.values()
    count = 0

    with open(DATAPATH + data_file, "wb") as f:
        for item in ['u1', 'u2', 'label',\
                    'cosine', 'euclidean', 'correlation', 'chebyshev', \
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']:
            f.write( ","+ item )
        f.write("\n")


    for i in pair.index:

        u1 = pair.loc[i, 'u1']
        u2 = pair.loc[i, 'u2']
        label = pair.loc[i, 'label']

        # does the user have an embedding
        if u1 in emb_uid and u2 in emb_uid:

            u1_vector = emb.loc[u1]
            u2_vector = emb.loc[u2]

            i_feature = pd.DataFrame([[u1, u2, label, \
                                       distance.cosine(u1_vector, u2_vector), \
                                       distance.euclidean(u1_vector, u2_vector), \
                                       distance.correlation(u1_vector, u2_vector), \
                                       distance.chebyshev(u1_vector, u2_vector), \
                                       distance.braycurtis(u1_vector, u2_vector), \
                                       distance.canberra(u1_vector, u2_vector), \
                                       distance.cityblock(u1_vector, u2_vector), \
                                       distance.sqeuclidean(u1_vector, u2_vector)]])

            i_feature.to_csv(DATAPATH + data_file, \
                             index=False, header=None, mode='a')

        else:
            print (u1, u2, "one of the pair not found")
