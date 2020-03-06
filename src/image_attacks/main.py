# Created by rahman at 15:20 2020-03-05 using PyCharm
import subprocess

from src.image_attacks.im_utils import slice_files, combine_files, clean_trim, count_cats, make_dataset_counts, score

from src.shared_utils.shared_tools import make_allPairs, classifiers



def attack_images():
    
    city = 'la'  # 'ny', sys.argv[1]
    cores = 120
    prob_cutoff = 0.05

    DATAPATH = '../../data/' + city
    mediaFile = "target_media"

    slice_files(mediaFile, DATAPATH, cores)

    subprocess.call(['./parallelize_im2proba.sh', cores,
                     city])  # downloads images and converts to embeddings, shell script calls im2proba.py

    prob_file = combine_files(DATAPATH, cores)

    clean_file = clean_trim(prob_cutoff, DATAPATH, prob_file)

    counts_file = count_cats(DATAPATH, clean_file)

    allPairs = make_allPairs("avg_pairs.csv", u_list_file=counts_file, DATAPATH=DATAPATH,
                             friendFile=city + ".target_friends", makeStrangers=True)

    dataset = make_dataset_counts(DATAPATH, clean_file, DATAPATH + "supervised_ensemble2/im_dataset.csv", counts_file,
                                  allPairs)

    score(dataset, name="mini-counts, cosine, entropy of max cat", classifiers=classifiers)


if __name__ == '__main__':

    attack_images()













