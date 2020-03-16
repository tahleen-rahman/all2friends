# Created by rahman at 17:20 2020-03-10 using PyCharm
import sys

from image_attacks.im_main import attack_images
from captions_attacks.cap_main import attack_captions
from hashtag_attacks.ht_main import attack_hashtags
from location_attacks.loc_main import attack_locations
from multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI, write_posteriors, \
    unite_posteriors, score_avg5probs, score_subsets_weighted, split_train_test_cv, split_train_test
from network_attacks.friend2vec_main import attack_network
from shared_tools.utils import  DATAPATH, city



i, monomodal = str(1), False #sys.argv[1], sys.argv[2]




if monomodal: # if not multimodal only

    cap_file = attack_captions(th=0.01, sublinear=True)

    ht_file = attack_hashtags(th=0.001)

    im_file = attack_images(cores = 120, prob_cutoff = 0.05)

    # get all pairs which have atleast 1 among image, ht or caption data
    pairs, cap_dataset, ht_dataset, im_dataset = makeHCI(DATAPATH, 'HCI.csv',  im_file, ht_file, cap_file)

    # try to recalculate missing data for pairs that we have missed from one of the modalities but could have calculated.
    cap_file, ht_file, im_file = recalculate_missingHCI(DATAPATH, pairs, cap_dataset, ht_dataset, im_dataset)

    loc_file = attack_locations()


else:

    cap_file, ht_file, im_file, loc_file = "extra_cap_dataset.csv", "extra_ht_dataset.csv", "extra_im_dataset.csv", "loc_dataset.csv"


# get the file suffixes for each cross val iteration subgraph of friends in the training set
# to use for node2vec for the network attack as well as for the multimodal attack later
friends_train_file = split_train_test_cv(DATAPATH, i)

network_file = attack_network(friends_train_file, i)

write_posteriors(cap_file, ht_file, im_file, loc_file, network_file, DATAPATH, i)

unite_posteriors(DATAPATH, i)

score_avg5probs(DATAPATH, i)

score_subsets_weighted(DATAPATH, i)
