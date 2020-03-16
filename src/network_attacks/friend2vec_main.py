# Created by rahman at 16:50 2020-03-10 using PyCharm
from multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI, split_train_test
from network_attacks.friend2vec_utils import friend2vec, make_features_distances, make_features_hada
from shared_tools.utils import classifiers, DATAPATH, city


def attack_network(frn_train, i=""):
    """

    :return:
    """
    # get the incomplete subgraph for Random walk to get embeddings

    friend2vec(DATAPATH, frn_train, i)  # remember to delete old .walk file

    data_file = make_features_hada(DATAPATH, i)

    data_file_dist = make_features_distances(DATAPATH, i)

    print ("Created network datasets at", data_file, data_file_dist )

    return data_file


if __name__ == '__main__':

    friends_train_file = split_train_test(DATAPATH)

    data_file = attack_network(friends_train_file)



