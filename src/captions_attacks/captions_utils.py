# Created by rahman at 15:39 2020-03-09 using PyCharm


import os
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,braycurtis, canberra, cityblock, sqeuclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse
from sklearn.metrics import roc_auc_score



def make_dfwords(city, DATAPATH):
    """
    joins all captions_results for each user
    :param city:
    :param DATAPATH:
    :return:
    """
    caption = pd.read_csv(DATAPATH + city + ".target_caption")

    caption['caption'] = caption['caption'].str.lower()

    #collect all captions for each user
    grouped = caption.groupby('uid')
    dataarr = []
    for uid, group in grouped:
        caplist = group.caption
        capstr = ' '.join(map(str, caplist))
        dataarr.append([uid, capstr])
        # print len(capstr)

    df = pd.DataFrame(data=dataarr, columns=['uid', 'words'])

    df.to_csv(DATAPATH + "words.csv", index=False)

    # df=pd.read_csv(DATAPATH + "words.csv")
    print ("joined all captions_results for each user", df.shape)

    return df




def get_TFIDF_filtered(sublinear,th, df, DATAPATH, npzBOWfile):
    """
    Convert a collection of  words to a matrix of TF-IDF features.
    :param sublinear:  boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    :param th: When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific stop words).
    :param df: df containiing all words per user
    :param DATAPATH:
    :param npzBOWfile: file containing sparse TFIDF matrix
    :return:
    """

    vectorizer = TfidfVectorizer(max_df=th, sublinear_tf=sublinear, min_df=2) # CountVectorizer(min_df=2)

    tfidf_matrix=vectorizer.fit_transform(df.words)

    scipy.sparse.save_npz(DATAPATH + npzBOWfile, tfidf_matrix)

    print ("created ", npzBOWfile)
    print ("tfidf_matrix.shape", tfidf_matrix.shape)

    # get idf for each word
    wordids = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    wordidDF = pd.DataFrame(wordids.items(), columns=['word', 'tf_idf'])

    wordidDF.to_csv(DATAPATH +str(sublinear)+ str(th)+"_tfids.csv")

    print ("created" +str(sublinear)+ str(th)+"_tfids.csv")
    print ("number of words len(vectorizer.idf_)",  len(vectorizer.idf_))

    assert(df.shape[0] == tfidf_matrix.shape[0])

    return tfidf_matrix



def make_features(dataFile, df,  TFIDF_matrix,  pairs, DATAPATH):

    """

    :param dataFile: output features in this file
    :param df: df of users and words
    :param TFIDF_matrix: tfidf matrix of each user
    :param pairs:
    :param DATAPATH:
    :return:
    """

    print ("pairs.shape", pairs.shape, pairs.columns)

    if os.path.exists(DATAPATH + dataFile):
        print ("datafile exists, removing", dataFile)
        os.remove(DATAPATH + dataFile)


    count=0

    with open(DATAPATH + dataFile, "wb") as f:
        for item in ['u1', 'u2', 'label',\
                    'cosine', 'euclidean', 'correlation', 'chebyshev', \
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']:
            f.write( ","+ item )
        f.write("\n")


    for i in range(len(pairs)):

        if not (i % 500):
            print (i , "out  of ", len(pairs))
        label = pairs.loc[i, 'label']

        try:

            u1, u2 = pairs.loc[i, 'u1'], pairs.loc[i, 'u2']

            # retrieve the index of the user from the df containing words
            pos1ind, pos2ind = df[df.uid == u1].index, df[df.uid == u2].index

            pos1arr, pos2arr = pos1ind.get_values(), pos2ind.get_values()

            pos1, pos2 = pos1arr[0], pos2arr[0] # these 2 are still

            # and use the index to get the correct row from the tfidf matrix
            u1_vector, u2_vector = TFIDF_matrix[pos1, :].toarray(), TFIDF_matrix[pos2, :].toarray()

            i_feature = pd.DataFrame([[u1, u2, label, \
                                       cosine(u1_vector, u2_vector), \
                                       euclidean(u1_vector, u2_vector), \
                                       correlation(u1_vector, u2_vector), \
                                       chebyshev(u1_vector, u2_vector), \
                                       braycurtis(u1_vector, u2_vector), \
                                       canberra(u1_vector, u2_vector), \
                                       cityblock(u1_vector, u2_vector), \
                                       sqeuclidean(u1_vector, u2_vector)]])

            i_feature.to_csv(DATAPATH + dataFile, index=False,  header=None, mode='a')
            # print "feature created"

        except Exception as e:

            print ("EXCEPTION!",  u1, u2, e.message)
            count+=1

    print  (count , " pairs not found out of ",  len(pairs))





def score_all_aucs(dataFile, classifiers,  DATAPATH):
    """

    :param dataFile:
    :param classifiers:
    :param DATAPATH:
    :return:
    """

    dataset = pd.read_csv(DATAPATH+dataFile,  error_bad_lines=False)#names = ['u1', 'u2', 'label', \
             #'cosine', 'euclidean', 'correlation', 'chebyshev', \
            # 'braycurtis', 'canberra', 'cityblock', 'sqeuclidean'],

    print ("before dropna", dataset.shape)
    dataset.drop(dataset[dataset.label!=0][dataset.label!=1].index, inplace=True)

    for col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col])#, errors='coerce')

    dataset.dropna(inplace=True)
    print ("after drop na and bad lines", dataset.shape)

    auc_res = []
    for i in ['cosine', 'euclidean', 'correlation', 'chebyshev',\
              'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']:
        i_auc = roc_auc_score(dataset.label, dataset[i])
        if i_auc < 0.5: i_auc = 1-i_auc

        print (i, i_auc)
        auc_res.append(i_auc)




    for cname, classifier in classifiers.items():

        print("scoring  with", cname)

        clf = classifier[0](**classifier[1])
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        X = dataset.iloc[:, 3:].values
        Y = dataset.iloc[:, 2].values

        rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)

        for scorer in ['roc_auc']:  # 'accuracy', , 'average_precision'

            scores = cross_val_score(clf, X, Y, scoring=scorer, cv=rskf)
            print(scorer, " %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


