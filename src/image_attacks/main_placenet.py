import sys, pandas as pd
from im_utils import getProbsSequential, getProbsParallel, folder_setup, restart_ProbsSequential

city = 'la'
folder_setup(city)
DATAPATH ="../data/"+city+"/"


slice = sys.argv[1]

mediaFile = DATAPATH + str(slice)+ "media_cleaned.csv"

media = pd.read_csv(mediaFile, header=None, names=[u'mid', u'uid', u'uname', u'time', u'like', u'comment',u'url', u'code'])

getProbsSequential(media, DATAPATH, probFile=DATAPATH + str(slice) +"probability_dist.csv")

#restart_ProbsSequential(media, DATAPATH, probFile=DATAPATH +str(slice) +"probabilityDist.csv")
