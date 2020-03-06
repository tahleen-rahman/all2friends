import sys, pandas as pd

from src.image_attacks.im_utils import getProbsSequential, getProbsParallel,  restart_ProbsSequential
from src.shared_utils.shared_tools import folder_setup

slice, city  = sys.argv[1], sys.argv[2]

folder_setup(city)

DATAPATH = "../data/" + city+"/"
mediaFile = DATAPATH + str(slice)+ "media_cleaned.csv"

media = pd.read_csv(mediaFile, header=None, names=[u'mid', u'uid', u'uname', u'time', u'like', u'comment',u'url', u'code'])

getProbsSequential(media, DATAPATH, probFile=DATAPATH + str(slice) +"probability_dist.csv")

#restart_ProbsSequential(media, DATAPATH, probFile=DATAPATH +str(slice) +"probabilityDist.csv")
