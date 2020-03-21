# Created by rahman at 17:20 2020-03-10 using PyCharm
import sys

from image_attacks.im_main import attack_images
from captions_attacks.cap_main import attack_captions
from hashtag_attacks.ht_main import attack_hashtags
from location_attacks.loc_main import attack_locations
from multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI, write_posteriors2, \
    unite_posteriors, score_avg5probs, score_subsets_weighted, split_train_test_cv, split_train_test, make_results
from network_attacks.friend2vec_main import attack_network
from shared_tools.utils import  DATAPATH, city




cap_file, ht_file, im_file, loc_file = "extra_cap_dataset.csv", "extra_ht_dataset.csv", "extra_im_dataset.csv", "loc_dataset.csv"

        
network_file = attack_network(frn_train='train_pairs.csv')
    
write_posteriors2(cap_file, ht_file, im_file, loc_file, network_file, DATAPATH)
    

