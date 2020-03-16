# Created by rahman at 15:30 2020-03-09 using PyCharm

from shared_tools.utils import make_allPairs, classifiers, city, DATAPATH
from captions_attacks.captions_utils import  make_features, score_all_aucs,  make_dfwords, get_TFIDF_filtered


def attack_captions(th, sublinear):
    """

    :param th: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific stop words).
    :param sublinear:  boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    :return:
    """


    df = make_dfwords(city, DATAPATH)
    #df=pd.read_csv(DATAPATH + "words.csv")

    TFIDF_matrix = get_TFIDF_filtered(sublinear, th, df, DATAPATH, TFIDFfile = str(sublinear)+str(th)+'filt_TFIDF.npz')
    #TFIDF_matrix = scipy.sparse.load_npz(DATAPATH+ TFIDFfile)

    allPairs = make_allPairs(str(th)+str(sublinear)+"filt_TFIDF_pairs.csv",
                             u_list=df.uid,
                             DATAPATH=DATAPATH,
                             friendFile=city + ".target_friends",
                             makeStrangers=True)
    #allPairs=pd.read_csv(DATAPATH+str(th)+"filt_TFIDF_pairs.csv", index_col=0)


    data_file = "cap_dataset.csv" #no save

    make_features(data_file, df, TFIDF_matrix,  allPairs, DATAPATH)

    score_all_aucs(data_file, DATAPATH) #, classifiers=classifiers)

    print ("Created caption dataset at", data_file)

    return data_file




if __name__ == '__main__':

    data_file = attack_captions(th=0.01, sublinear=True)

