# Created by rahman at 13:50 2020-03-05 using PyCharm

import sys
import requests
from scipy.spatial.distance import cosine
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from torch.autograd import Variable as V
from torch.nn import functional as F
import os, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

from image_attacks.run_placesCNN_unified import load_model, returnTF


def slice_files(mediaFile, DATAPATH, cores):
    """
    slices our big file target_media into many slices for parallel processing of the image urls later
    :param mediaFile: big files containing all the media samples
    :param DATAPATH:
    :param cores: number of cores in the compute server that I want to use
    :return:
    """

    with open(mediaFile, "r") as big_file:

        lines = big_file.readlines()
        slice_size = int(len(lines) / cores)

        for slice in range(0, cores):

            file_slice = lines[slice * slice_size:(slice + 1) * slice_size]
            print (slice * slice_size, (slice + 1) * slice_size)

            newfile = DATAPATH + str(slice) + "media_cleaned.csv"

            with open(newfile, 'w') as small_file:
                small_file.writelines(file_slice)

        rem_idx =  (slice + 1) * slice_size  # first index of the remaining lines

    # leftover remaining samples in cores + "media_cleaned.csv"
    with open(mediaFile, "r") as big_file:

        lines = big_file.readlines()

        with open(DATAPATH + cores + "media_cleaned.csv", 'w') as small_file:

            small_file.writelines(lines[rem_idx:]) #8309400


def combine_files(DATAPATH, cores):
    """
    cobmines the image embeddings created by the parallel shell scripts back into 1 large file
    :param DATAPATH:
    :param cores:
    :return:
    """

    combi_file, i = "bigger_combi_probfile.csv", 0

    with open(DATAPATH + combi_file, 'w') as big_file:

        for slice in range(0, cores):

            sliced_file = DATAPATH + str(slice) + "probability_dist.csv"

            with open(sliced_file, 'r') as small_file:

                file_slice = small_file.readlines()
                print (len(file_slice))

                # write the header
                if i == 0:
                    big_file.write(file_slice[0])

                big_file.writelines(file_slice[1:])
                i+=1

    return combi_file


def get_prob_dist(img_name):

    ''' convert image to embeddings

    Args:
        img_name: the name of the target image
    Returns:
        img_proba: a numpy array of 365 array 1X365

    '''

    # load the model
    model = load_model()

    # load the transformer
    tf = returnTF()  # image transformer


    # load the test image
    img = Image.open(img_name)
    input_img = V(tf(img).unsqueeze(0), volatile=True)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit).data.squeeze()
    # print len(logit)
    img_proba = h_x.numpy()

    return img_proba


def write_distributions(cols, quadruple, probFile, DATAPATH):

    ''' downloads image from given url and gets probability distributions for an image from PlaceNET and writes to probFile
     Args:
         cols: column names for the probability file,
         quadruple: (uid, mid, url, index) from MediaCleaned.csv
         probFile: output file containing [uid, mid, prob_0..prob_364]
         DATAPATH: datapath
     Returns:
         None
     '''

    uid, mid, link = quadruple[0], quadruple[1], quadruple[2]

    if not os.path.exists(DATAPATH + "images/"):
        os.makedirs(DATAPATH + "images/")
    imageName = DATAPATH + "images/" + link.split('/')[-1]

    try:
        if not os.path.exists(imageName):
            #print "downloading ", imageName
            imgFile = open(imageName, "wb")
            imgFile.write(requests.get(link).content)
            imgFile.close()

    except:
        print ("could not download", imageName, link, link.split('/')[-1])
        return None

    try:
        cleanVect = get_prob_dist(imageName)

    except Exception as e:
        print ("could not vectorize , deleting ", imageName)

        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)

        os.remove(imageName)
        return None

    try:
        row=[uid, mid]
        row.extend(cleanVect)
        mediaVector=pd.DataFrame(data=[row], columns=cols)
        mediaVector.to_csv(probFile, mode="a", header=False)

    except Exception as e:
        print ("could not write vector to file ", imageName)

        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)

        os.remove(imageName)
        return None

    try:
        os.remove(imageName)
        print ("vector written to file, image deleted index: ", quadruple[3])

    except:
        print ("image NOT deleted or something weird happened", imageName)


def getProbsSequential(media, DATAPATH, probFile):

    ''' get probabilities sequentially from mids in mediaDF,
     Args:
         media: mediaDF
         probFile: output file containing [uid, mid, prob_0..prob_364]
         DATAPATH: datapath
     Returns:
         None
     '''

    # create column names
    cols = ["uid", "mid"]
    for i in range(0, 365):
         name = "prob_" + str(i)
         cols.append(name)

    # write column names at the top of file
    with open(probFile, "wb") as f:
        for item in cols:
            f.write( ","+ item )
        f.write("\n")


    for i in media.index:
        write_distributions(cols, [media.at[i, "uid"], media.at[i, "mid"], media.at[i, "url"], i ],probFile, DATAPATH)



def restart_ProbsSequential(media, DATAPATH, probFile):

    ''' get probabilities sequentially from mids in mediaDF,
     Args:
         media: mediaDF
         probFile: output file containing [uid, mid, prob_0..prob_364]
         DATAPATH: datapath
     Returns:
         None
     '''

    proba = pd.read_csv(probFile, error_bad_lines=False, index_col=0)
    proba.dropna(inplace=True)

    for col in proba.columns:
        proba[col] = pd.to_numeric(proba[col], errors='coerce')

    proba.dropna(inplace=True)
    proba.reset_index(inplace=True, drop=True)

    dropInds = [i for i in range(0, len(proba)) if sum(proba.iloc[i, 2:-1]) > 1.1 or sum(proba.iloc[i, 2:-1]) < 0.99]
    print ("len(dropInds)", len(dropInds))
    proba.drop(proba.index[dropInds], inplace=True)

    smallmedia = media.loc[-media.mid.isin(proba.mid.values)]
    print ("smallmedia.shape", smallmedia.shape)

    cols = ["uid", "mid"]

    for i in range(0, 365):
         name = "prob_" + str(i)
         cols.append(name)

    for i in smallmedia.index:
        write_distributions(cols, [smallmedia.at[i, "uid"], smallmedia.at[i, "mid"], smallmedia.at[i, "url"], i ],probFile, DATAPATH)


def getProbsParallel(media2, probFile, DATAPATH):

    """

    :param media2:
    :param probFile:
    :param DATAPATH:
    :return:
    """
    cols = ["uid", "mid"]
    for i in range(0, 365):
         name = "prob_" + str(i)
         cols.append(name)
    with open(probFile, "wb") as f:
        for item in cols:
            f.write( ","+ item )
        f.write("\n")

    """allPairs = pd.read_csv(DATAPATH + "allPairs.csv", index_col=0)
    small_pairs=allPairs[100:-100] #.append(allPairs[-100:])
    users=small_pairs.u1.append(small_pairs.u2, ignore_index=True)
    users.drop_duplicates(inplace=True)
    print "len(users)", len(users)"""
    #smallmedia=media.loc[media.uid.isin(users)]

    media = pd.read_csv(DATAPATH + "probabilityDist2.csv", error_bad_lines=False, index_col=0)
    media.dropna(inplace=True)

    for col in media.columns:
        media[col] = pd.to_numeric(media[col], errors='coerce')

    media.dropna(inplace=True)
    media.reset_index(inplace=True, drop=True)

    dropInds = [i for i in range(0, len(media)) if sum(media.iloc[i, 2:-1]) > 1.1 or sum(media.iloc[i, 2:-1]) < 0.99]
    print ("len(dropInds)", len(dropInds))
    media.drop(media.index[dropInds], inplace=True)
    media.reset_index(drop=True)
    print ("media.shape", media.shape, "media2.shape", media2.shape)

    smallmedia=media2.loc[media2.mid.isin(media.mid.values)]
    print ("len(smallmedia.uid.unique()), smallmedia.shape", len(smallmedia.uid.unique()), smallmedia.shape)

    Parallel(n_jobs = 120) (delayed(write_distributions)(cols,[media2.at[i,"uid"], media2.at[i,"mid"], media2.at[i,"url"], i], probFile, DATAPATH) for i in smallmedia.index)


def clean_trim(cutoff, DATAPATH, probFile):
    """
    cleans probabillities and trims small values

    :param cutoff:
    :param DATAPATH:
    :param probFile:
    :param cleanFile:
    :return:
    """

    cleanFile = str(cutoff) + "proba_cut.csv"

    with open(DATAPATH + "BADLINES_probs", "wb") as fp:
        # some lines are not the same number of columns bcz of parallel writing maybe
        sys.stderr = fp
        proba = pd.read_csv(DATAPATH + probFile, error_bad_lines=False, index_col=0)

    print ("proba.shape", proba.shape)
    proba.dropna(inplace=True)

    # remove rows and cols that could not be converted to numbers, sometimes there was multiple decimal points
    for col in proba.columns:
        proba[col] = pd.to_numeric(proba[col], errors='coerce')
    proba.dropna(inplace=True)
    proba.reset_index(inplace=True, drop=True)


    # probabilities should sum to 1 TODO check floating point precision
    dropInds = [i for i in range(0, len(proba)) if sum(proba.iloc[i, 2:-1]) >= 1.1 or sum(proba.iloc[i, 2:-1]) < 0.99]
    print ("len(dropInds)", len(dropInds))
    proba.drop(proba.index[dropInds], inplace=True)
    proba.reset_index(drop=True)


    #proba.loc[:, "entropy"] = entropy(proba.iloc[:, 2:].T.astype(pd.np.float64).values.tolist())

    proba[proba < cutoff] = 0

    print ("trimmed small values")

    proba.to_csv(DATAPATH + cleanFile)

    return cleanFile




def count_cats(DATAPATH, cleanFile, countsFile ):
    """
    counts the number of images in each category for a user
    :param DATAPATH:
    :param cleanFile: cleaned probability file
    :return: count file
    """

    proba = pd.read_csv(DATAPATH + cleanFile, index_col=0)

    grouped = proba.groupby('uid')
    counts = grouped.apply(lambda col: (col != 0).sum())

    counts.to_csv(DATAPATH + countsFile)#, index=False)
    #counts = pd.read_csv(DATAPATH + countsFile, index_col=0)

    print(counts.columns)
    print(counts.index)

    #
    counts.drop(counts.columns[0], axis=1, inplace=True)

    # we dont want the counts of mid
    counts.drop(['mid'], axis=1, inplace=True)
    # counts.set_index('uid', drop=True, inplace=True)

    print(counts.columns)
    print(counts.index)

    counts.to_csv(DATAPATH + countsFile)

    return countsFile


def makeEntropies(DATAPATH, proba):
    """
    We perform a category-wise sum of the posterior probabilities over all images shared by each user which we represent by a
    usage-vector where each element represents the summed up probability that a category c is depicted over all images shared by uj .
    :param DATAPATH:
    :param proba:
    :return:
    """
    data_ent = []
    proba_round = proba

    for cat in range(2, len(proba_round.columns)):
        sums = proba_round.groupby('uid')[proba_round.columns[cat]].agg(pd.np.sum)
        e = entropy(sums)
        data_ent.append([proba_round.columns[cat][5:], e])

    ent_df = pd.DataFrame(data=data_ent, columns=["category", "entropy"])
    ent_df.to_csv(DATAPATH  + "entropy_of_categories.csv")
    return ent_df


def make_features_counts(DATAPATH, clean_file, dataFile, counts_file,  allPairs):
    """
    calculates our features for each user pair from the counts vector of each user ,   our features are
    the number of times each category is shared by both users in the pair + maximum index + max counts + entropy
    refer to paper for details

    :param DATAPATH:
    :param proba: the original probability distributions for all images, need this for entropy
    :param dataFile: output file containing features for all pairs
    :param counts: counts file
    :param allPairs: true and false user pairs of our dataset
    :return: dataFile
    """


    proba = pd.read_csv(DATAPATH + clean_file, index_col=0)

    counts = pd.read_csv(DATAPATH + counts_file, index_col=0)

    dataset = allPairs
    print ("making dataset file ", dataFile, " from ", len(allPairs), " pairs")

    dataset['cosine_Dist']=-99
    print ("adding column cosine", len(dataset.columns))

    # initialize all columns for the pairs
    for i in range(0, len(counts.columns)):
        dataset["comp_" + str(i)] = -99
    print (len(counts.columns), "columns set to -99")

    # first cosine distance and minumums
    count=0
    for row in dataset.index:

        try:
            counts1 = counts.loc[dataset.loc[row, 'u1']]
            counts2 = counts.loc[dataset.loc[row, 'u2']]

            dataset.loc[row, 'cosine_Dist'] = cosine(counts1, counts2)

            #  the number of times each category is shared by both users in the pair, so minimum
            mini = pd.np.minimum(counts1, counts2)
            dataset.iloc[row, -len(counts.columns):] = mini.values

        except Exception as e:
            count+=1

    print ("dataset.shape, dataset.dropna().shape",  dataset.shape, dataset.dropna().shape)

    #  more features: frequency of the mutually most frequently shared category in images shared by both users
    maxcounts = dataset.iloc[:,4:].max(axis=1)
    maxcats = dataset.iloc[:,4:-1].idxmax(axis=1)
    dataset['max_count'] = maxcounts
    dataset['maxcat'] = maxcats

    # last feature: entropy of the usage vector of the mutually most frequently shared category
    ent_df =  makeEntropies(DATAPATH, proba)
    dataset['ent_maxcat'] =  [ent_df.loc[int(cat[5:])].entropy for cat in dataset.maxcat.values]

    dataset = dataset[dataset.max_count!=-99]
    print ("after dropping -99 rows", dataset.shape)

    #dataset=remove_for_Entropy(dataset)
    dataset.dropna(inplace=True)

    frns = dataset[dataset.label == 1]
    strangers = dataset[dataset.label == 0]

    print ("after drop na, dataset.shape, pairs not found in countsDF (should be the no of -99 rows dropped)", dataset.shape, count)
    print (" len(frns), len(strangers)",  len(frns), len(strangers))


    dataset.drop(['maxcat'], inplace=True, axis=1) #  I remove this bcz for some reason i forget

    dataset.to_csv(dataFile, index=False)

    return dataset


def score(dataset, name, classifiers):

    """
    trains classifier and calculates 'accuracy', 'roc_auc', 'average_precision' with n_splits=3, n_repeats=10
    for concat, repeats 5 times
    prints std output

    :param dataset: header of the form [u1,u2,label,cosine_Dist,comp_0, ... comp_364 ]
    :param name: name of the similarity metric ('L2': L2,  'AVG':AVG, 'L1':L1,'HADAMARD': HADAMARD, 'concat':concat )
    :param classifiers: {'RF':(RandomForestClassifier, {"n_estimators": 101,  "max_depth": 10})}#,
             #'GBM': (GradientBoostingClassifier,{'n_estimators':100, 'max_depth': 3}),
             #'LR_SAG_L2penalty':(LogisticRegression, {'solver': 'sag'}),
             #'LR_liblinear_L2penalty': (LogisticRegression, {'solver': 'liblinear', 'penalty': 'l2'})}
    :return: None
    """
    print ("cosine_Dist", 1 - roc_auc_score(dataset.label, dataset.cosine_Dist))
    print ("max_count", 1 - roc_auc_score(dataset.label, dataset.max_count))
    print ("ent_maxcat", 1 - roc_auc_score(dataset.label, dataset.ent_maxcat))


    print ("scoring " , dataset.columns[3:-3])

    for cname, classifier in classifiers.items():

        print ("scoring only 365 count features with", cname)

        clf = classifier[0](**classifier[1])

        dataset = dataset.sample(frac=1).reset_index(drop=True)
        X = dataset.iloc[:, 3:-3].values
        Y = dataset.iloc[:, 2].values

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

        for scorer in ['roc_auc']:#'accuracy', , 'average_precision'

            scores = cross_val_score(clf, X, Y, scoring=scorer, cv=rskf)
            print(scorer, " %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


    print ("scoring ", dataset.columns)

    result, names = [], []
    for cname, classifier in classifiers.items():

        print ("scoring all features with", cname)

        clf = classifier[0](**classifier[1])

        dataset = dataset.sample(frac=1).reset_index(drop=True)
        X = dataset.iloc[:, 3:].values
        Y = dataset.iloc[:, 2].values

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

        for scorer in [ 'roc_auc']:#'accuracy',, 'average_precision'

            scores = cross_val_score(clf, X, Y, scoring=scorer, cv=rskf)

            print(scorer, " %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

            result.append(scores.mean())
            names.append(cname)

    return result, names

