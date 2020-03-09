# Created by rahman at 12:23 2020-03-09 using PyCharm
from src.hashtag_attacks.hashtag2vec_utils import uh_graph_build, make_walk, emb_train, feature_construct, \
    predict_all_aucs
from src.hashtag_attacks.hashtag_utils import filter_hashtags_users, make_features, score
from src.shared_utils.shared_tools import classifiers, DATAPATH, city, pair_construct



def attack_hashtags(th, do_hashtag2vec=True):

    """

    first use hand-enginnered features, then use embeddings if do_hashtag2vec=true
    hashtags are too popular if more than th% of users share them
    :return:
    """


    entropy_df, ht = filter_hashtags_users(DATAPATH , th, city)

    datafile = str(th) + "_dataset.csv"

    pairs = pair_construct(ht.uid.unique(), friendFile=city + ".target_friends", downSample=False)

    dataset = make_features(th, ht, entropy_df, datafile, pairs)

    #dataset = pd.read_csv(DATAPATH+datafile)

    score(dataset, classifiers, UNSUP=True)

    # use embeddings from a random walk over user-hashtag biprtite graph
    if do_hashtag2vec:

        model_name = str(th) + "_" + city

        uh_graph, hu_graph = uh_graph_build(ht)

        make_walk( DATAPATH,uh_graph, hu_graph, model_name, core_num=mp.cpu_count()-1)

        emb_train(  DATAPATH,model_name)

        feature_construct(DATAPATH, model_name, pairs)

        predict_all_aucs(DATAPATH, model_name)





if __name__ == '__main__':

    for th in [0.001]:   #, 0.005 ,0.01, 0.05 ,0.1]:

        result, names = attack_hashtags(th)




