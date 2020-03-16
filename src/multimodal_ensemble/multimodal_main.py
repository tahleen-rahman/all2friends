# Created by rahman at 17:20 2020-03-10 using PyCharm
import sys


from multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI, write_posteriors, \
    unite_posteriors, score_avg5probs, score_subsets_weighted, split_train_test_cv, split_train_test
from network_attacks.friend2vec_main import attack_network
from shared_tools.utils import  DATAPATH, city



i = sys.argv[1]



cap_file, ht_file, im_file, loc_file = "extra_cap_dataset.csv", "extra_ht_dataset.csv", "extra_im_dataset.csv", "loc_dataset.csv"

# get the file suffixes for each cross val iteration subgraph of friends in the training set
# to use for node2vec for the network attack as well as for the multimodal attack later
friends_train_file = split_train_test_cv(DATAPATH, i)

network_file = attack_network(friends_train_file, i)

write_posteriors(cap_file, ht_file, im_file, loc_file, network_file, DATAPATH, i)

unite_posteriors(DATAPATH, i)

score_avg5probs(DATAPATH, i)

score_subsets_weighted(DATAPATH, i)
