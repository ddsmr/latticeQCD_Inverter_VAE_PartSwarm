import matplotlib
matplotlib.use('Agg')
import numpy as np
np.set_printoptions(linewidth=1000)
import pickle
from scipy.sparse.linalg import *
import sys
import matplotlib.pyplot as plt
# sys.path.insert(1,'../')
# sys.path.insert(1,'../../')
for insertPath in ['../', '../../', '../../examples', 'Utils']:
    sys.path.insert(1, insertPath)
from latmattools.manip import manip
from latmattools.algs import conjgrad
import random

Ls = 2
Lt = 2

# source vector setup
source = np.ones(3*Lt*Ls**3,dtype=np.cdouble)

# load predicted inverses
predsraw = pickle.load(open('../Predictions/predInvMats_DetMin.p','rb'))
print(predsraw.keys(), len(predsraw['InvPreds_Im']))
nbMats = len(predsraw['InvPreds_Im'])

preds = predsraw['InvPreds_Re'] + 1j*predsraw['InvPreds_Im']

print(type(preds))
# exit()
# preds = np.zeros((1972,48,48),dtype=np.cdouble)
# goodchisq = []
# allchisq = []
# for i in range(1972):
#     allchisq.append(predsraw[str(i)]['Chi2'])
#     if predsraw[str(i)]['Chi2'] < 10:
#         #print(predsraw[str(i)]['Chi2'])
#         goodchisq.append(i)
#     #print(predsraw[str(i)]['Mat'])
#     for j in range(48):
#         for k in range(48):
#             preds[i][j][k] = predsraw[str(i)]['Mat'][j][k]

# apply random mask for validation set based on what was used for training set
np.random.seed(0)
matArray = pickle.load(open('../../examples/picklejar/allmats-10000', 'rb'))
invMatArray = pickle.load(open('../../examples/picklejar/allinvmats-10000', 'rb'))
matDict = {'matArray': matArray,
           'invMatArray': invMatArray
           }
trainSplit = 0.85  # zSpace['TrainSplit']

#print(preds[0])

boolMask = np.array([True if np.random.uniform() < trainSplit else False
                     for _ in range(matDict['matArray'].shape[0])
                     ])
mats = matDict['matArray'][np.logical_not(boolMask)]
invmats = matDict['invMatArray'][np.logical_not(boolMask)]

from trainVAEs import splitMatArrs
trainArray, mats = splitMatArrs(matDict['matArray'],  0.85)

#print(invmats[0])
# print(np.subtract(preds[0],invmats[0]))

cgtest = conjgrad.CGTest(preds,mats[0:nbMats],source)
cgtest()
cgtest.make_plots()


print(goodchisq)
#print(allchisq)

for i in goodchisq:
    print(cgtest.itercounts[i] - cgtest.itercountspred[i])
