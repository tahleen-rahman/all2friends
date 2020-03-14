# Created by rahman at 15:20 2020-03-05 using PyCharm
import subprocess

from src.image_attacks.im_utils import slice_files, combine_files, clean_trim, count_cats, make_features_counts, score

from src.shared_tools.utils import make_allPairs, classifiers, DATAPATH, city



def attack_images(cores, prob_cutoff):
    """

    :param cores: how many cores to use for multiprocessing
    :param prob_cutoff: user's image belongs to a certain category if the output of the last FC layer of the resnet model for the category  > prob_cutoff
    :return:
    """


    mediaFile = "target_media"

    slice_files(mediaFile, DATAPATH, cores)

    subprocess.call(['./parallelize_im2proba.sh', cores,
                     city])  # downloads images and converts to embeddings, shell script calls im2proba.py

    prob_file = combine_files(DATAPATH, cores)

    clean_file = clean_trim(prob_cutoff, DATAPATH, prob_file)

    counts_file = count_cats(DATAPATH, clean_file, countsFile="proba_cut_01_counts.csv" )

    allPairs = make_allPairs("avg_pairs.csv", u_list_file=counts_file, DATAPATH=DATAPATH,
                             friendFile=city + ".target_friends", makeStrangers=True)

    data_file = DATAPATH + "im_dataset.csv"

    dataset = make_features_counts(DATAPATH, clean_file, data_file, counts_file,
                                  allPairs)

    score(dataset, name="mini-counts, cosine, entropy of max cat", classifiers=classifiers)

    print ("Created image dataset at", data_file)

    return data_file


if __name__ == '__main__':

    data_file = attack_images(cores = 120, prob_cutoff = 0.05)













