# Created by rahman at 17:20 2020-03-10 using PyCharm
from src.captions_attacks.cap_main import attack_captions
from src.hashtag_attacks.ht_main import attack_hashtags
from src.image_attacks.im_main import attack_images
from src.location_attacks.loc_main import attack_locations
from src.multimodal_ensemble.multimodal_utils import makeHCI, recalculate_missingHCI, write_posteriors, unite_posteriors, score_avg5probs, score_subsets_weighted
from src.network_attacks.friend2vec_main import attack_network
from src.shared_tools.utils import  DATAPATH, city


attack_captions(th=0.01, sublinear=True)

attack_hashtags(th=0.001)

attack_images(cores = 120, prob_cutoff = 0.05)

# get all pairs which have atleast 1 among image, ht or caption data
pairs, cap_dataset, ht_dataset, im_dataset = makeHCI(DATAPATH)

# try to recalculate missing data for pairs that we have missed from one of the modalities but could have calculated.
cap_file, ht_file, im_file = recalculate_missingHCI(DATAPATH, pairs, cap_dataset, ht_dataset, im_dataset)

loc_file = attack_locations()

network_file = attack_network()

write_posteriors(cap_file, ht_file, im_file, loc_file, network_file, DATAPATH)


unite_posteriors(DATAPATH)

score_avg5probs(DATAPATH)

#score_subsets(DATAPATH)
score_subsets_weighted(DATAPATH)
