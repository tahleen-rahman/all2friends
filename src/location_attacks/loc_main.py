# Created by rahman at 11:58 2020-03-10 using PyCharm
from src.location_attacks.loc_utils import make_features

from src.shared_tools.utils import classifiers, DATAPATH, city
from src.multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI


def attack_locations():
    """
    using location embeddings from walk2freinds, make features to infer friendhsips
    :return:
    """


    data_file = make_features(city, DATAPATH, data_file="loc_dataset.csv")

    print ("Created location dataset at", data_file)

    return data_file




if __name__ == '__main__':

    loc_file = attack_locations()



