import numpy as np
import pickle
import argparse

from pprint import pprint as pp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import pyswarms as ps

import sys
for insertPath in ['../', '../examples', 'Utils']:
    sys.path.insert(1, insertPath)

from normArrays import ArrayNormaliser
from trainVAEs import loadAllMats, splitMatArrs


def functMin(x, origMat, decRe, decIm, n_particles, normDict, typeChiSq, matSize=48, typeNorm='SetMeanStd'):
    '''
        Define a chi square measure.
    '''
    reM, imM = origMat.real, origMat.imag

    reMinv = decRe.predict(x[:, 0:5]).reshape((n_particles, matSize, matSize))
    reMinv = normDict['ReNorm'](reMinv, typeNorm)

    imMinv = decIm.predict(x[:, 5:10]).reshape((n_particles, matSize, matSize))
    imMinv = normDict['ImNorm'](imMinv, typeNorm)

    chiSq_compRe = np.matmul(reM, reMinv[:]) - np.matmul(imM, imMinv) - np.identity(matSize)
    chiSq_compIm = np.matmul(reM, imMinv) + np.matmul(reMinv, imM)

    if typeChiSq == 'Det':
        # print('*******************************')
        reChiSq = (np.linalg.det(chiSq_compRe))**2
        imChiSq = (np.linalg.det(chiSq_compIm))**2

        return reChiSq + imChiSq
    elif typeChiSq == 'ColVec':
        # print('oooooooooooooooooooooooooo')
        def matmag(mat):
            mag = 0.
            for vec in mat:
                mag += np.sqrt(np.dot(vec, vec))
                return mag
        matMags = []
        for partNb in range(chiSq_compRe.shape[0]):
            mat = (chiSq_compRe + chiSq_compIm)[partNb]
            matMags.append(matmag(mat))
        matMags = np.array(matMags)
        # print(matMags.shape)
        # exit()
        return matMags
        # return matmag(chiSq_compRe) + matmag(chiSq_compIm)


def swarmMinChiSq(decRe, decIm, origMat, invMatRe_Normer, invMatIm_Normer, typeChiSq,
                  n_particles=15, nbIters=400):
    '''
    '''
    hyperParams = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 5, 'p': 2}

    normDict = {'ReNorm': invMatRe_Normer.invNormData,
                'ImNorm': invMatIm_Normer.invNormData}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=10, options=hyperParams)

    # Perform optimization; extract the bes =t cist and its posision
    cost, predZpos = optimizer.optimize(functMin, iters=nbIters, decRe=decRe, decIm=decIm, origMat=origMat,
                                        n_particles=n_particles, normDict=normDict, typeChiSq=typeChiSq)

    predZpos = np.array(predZpos).reshape((1, 10))

    matSize = 48
    reMinv = decRe.predict(predZpos[:, 0:5]).reshape((matSize, matSize))
    imMinv = decIm.predict(predZpos[:, 5:10]).reshape((matSize, matSize))

    return cost, reMinv, imMinv


def loadNormer(netStr):
    '''
        Load up the normalisation instance associated with the netStr.
    '''
    if 'zSpaceMaps' not in netStr:
        with open(netStr + '/VAE_Info.p', 'rb') as pcklIn:
            netAttrs = pickle.load(pcklIn)

        return netAttrs['runDict']['NormInstance'], netAttrs['runDict']['NormType']
    else:
        with open(netStr + '/Znorm.p', 'rb') as pcklIn:
            zNormInst = pickle.load(pcklIn)
        return zNormInst


if __name__ == '__main__':
    chiSqMethod = 'Det'
    # Load and split the matrices via train/test and Re/Im
    matDict = loadAllMats()
    trainArray, testArray = splitMatArrs(matDict['matArray'],  0.85)

    # Load up the decoders
    invMatVae_Re, invMatVae_Im = 'invMatArray-linArch_N500_L3_relu_tanh_Z5_Re', 'invMatArray-linArch_N1000_L3_relu_tanh_Z5_Im'
    invMatDecoder_Re = keras.models.load_model('VAEs/' + invMatVae_Re + '/Decoder')
    invMatDecoder_Im = keras.models.load_model('VAEs/' + invMatVae_Im + '/Decoder')

    invMatRe_Normer, normTy_Re = loadNormer('LatentZspaces/invMatArray_Zspace_' + invMatVae_Re)
    invMatIm_Normer, normTy_Im = loadNormer('LatentZspaces/invMatArray_Zspace_' + invMatVae_Im)

    nbTestMats = testArray.shape[0]
    predMats_Re, predMats_Im, chiSqList = [], [], []

    for matNb in range(5):
        origMat = testArray[matNb]
        chiSq, pred_Re, pred_Im = swarmMinChiSq(invMatDecoder_Re, invMatDecoder_Im, origMat,
                                                invMatRe_Normer, invMatIm_Normer, chiSqMethod)
        predMats_Re.append(pred_Re)
        predMats_Im.append(pred_Im)
        chiSqList.append(chiSq)

    predMats_Re = np.array(predMats_Re)
    predMats_Im = np.array(predMats_Im)

    predMats_Re = invMatRe_Normer.invNormData(predMats_Re, normTy_Re)
    predMats_Im = invMatIm_Normer.invNormData(predMats_Im, normTy_Im)

    predDict = {'ChiSqList': chiSqList, 'InvPreds_Re': predMats_Re, 'InvPreds_Im': predMats_Im}
    with open(f'Predictions/predInvMats_{chiSqMethod}Min.p', 'wb') as pcklOut:
        pickle.dump(predDict, pcklOut)
