# Created by rahman at 16:50 2020-03-10 using PyCharm
from src.multimodal_ensemble.multimodal_utils import split_train_test, makeHCI, recalculate_missingHCI
from src.network_attacks.friend2vec_utils import friend2vec, make_features_distances, make_features_hada
from src.shared_tools.utils import classifiers, DATAPATH, city


def attack_network():
    """

    :return:
    """
    # get the incomplete subgraph for Random walk to get embeddings
    frn_train = split_train_test(DATAPATH)

    friend2vec(DATAPATH, frn_train)  # remember to delete old .walk file

    data_file = make_features_hada(DATAPATH)

    data_file_dist = make_features_distances(DATAPATH)

    print ("Created network datasets at", data_file, data_file_dist )

    return data_file


if __name__ == '__main__':

    data_file = attack_network()



