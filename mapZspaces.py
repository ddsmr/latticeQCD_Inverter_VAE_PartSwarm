import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt


from pprint import pprint as pp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from keras.models import Sequential


import sys
for insertPath in ['../', '../examples', 'Utils']:
    sys.path.insert(1, insertPath)

from normArrays import ArrayNormaliser
from trainVAEs import loadAllMats, splitMatArrs, mkDir


def makeLinArch(inpDim, nbNeurons, nbLayers, outDim, actFct='relu', fitHandle='ZSpaceFit',
                lossFct='mean_squared_error', optType='adam', dropRate=0.2):
    '''
        Make a linear fully connected architecture with nbNeurons per layer, where the intemediate number of layers is
        specified by nbLayers, the dimension of the input and the output layers are specified by inpDim and outDim.
    '''
    # Start offf with the layer
    inputLayer = keras.Input(shape=(inpDim,))

    layerDict = {}
    dropDict = {}
    for layerNb in range(nbLayers):
        layerDict[str(layerNb)] = layers.Dense(nbNeurons, activation=actFct)
        dropDict[str(layerNb)] = layers.Dropout(dropRate)

    layerDict['0'] = layerDict['0'](inputLayer)
    dropDict['0'] = dropDict['0'](layerDict['0'])
    for layerNb in range(nbLayers - 1):
        # layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](layerDict[str(layerNb)])
        layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](dropDict[str(layerNb)])
        dropDict[str(layerNb + 1)] = dropDict[str(layerNb + 1)](layerDict[str(layerNb)])

    # outLayer = layers.Dense(outDim, activation=actFct)(layerDict[str(nbLayers - 1)])
    outLayer = layers.Dense(outDim, activation=actFct)(dropDict[str(nbLayers - 1)])

    model = keras.Model(inputs=inputLayer, outputs=outLayer, name=fitHandle)
    model.compile(loss=lossFct, optimizer=optType,
                  # metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
                  )
    model.summary()
    keras.utils.plot_model(model, fitHandle + ".png", show_shapes=True)

    return model


def initParser():
    '''
        Initialise and parse the arguments.
    '''
    parser = argparse.ArgumentParser(description='Process the inputs for the latent Z space mapping.')
    parser.add_argument('Re//Im', help='Input Re to train the real part, Im to train the imaginary part')

    parser.add_argument("-nE", '--nbOfEpochs', help='Specify number of training epochs.',
                        type=int, default=10000)
    parser.add_argument("-mB", '--mBatch', help='Specify minibatch size.',
                        type=int, default=128)
    parser.add_argument("-nNN", '--nbOfNeurons', help='Specify number of neurons per connected layer.', type=int, default=100)
    parser.add_argument("-nL", '--nbLayers', help='Specify number of connected layers.', type=int, default=1)
    parser.add_argument('--normData', help='Set flag to normalise data as either UnitVec or SetMeanStd', type=str, default='')
    parser.add_argument('--validSplit', help='Specify the validation split, default to 0.8.', type=float,
     default=0.2)
    parser.add_argument('--trainSplit', help='Specify the training split, default to 0.8.', type=float, default=0.8)
    argsPars = parser.parse_args()
    trainCard = vars(argsPars)

    return trainCard


def flattenArray(matArray):
    '''
        Flattens the matrix structure in the array.
    '''
    nbMatrices = matArray.shape[0]
    trainMats = []
    for i in range(nbMatrices):
        trainMats.append(matArray[i].flatten())
    trainMats = np.array(trainMats)

    return trainMats


def showStats(history):
    '''
        Show statistics related to the fitting.
    '''
    saveDir = 'AnalysisPlots/'

    # Show the training losses
    plt.plot(history.history['loss'], c='C0')
    plt.plot(history.history['val_loss'], c='C1')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(saveDir + 'trainLosses.pdf')
    plt.show()


if __name__ == '__main__':
    '''
        Main calls for the Z space fits.
    '''
    trainCard = initParser()
    commRoot_Mat = 'LatentZspaces/matArray_Zspace_matArray-'
    commRoot_Inv = 'LatentZspaces/invMatArray_Zspace_invMatArray-'
    vaeStr = '/VAE_Info.p'

    # Matrix Z space declarations
    zSpace_ReM, zSpace_ImM = 'linArch_N300_L3_relu_tanh_Z5_Re', 'linArch_N300_L3_relu_tanh_Z5_Im'
    # Inverse Matrix Z space declarations
    zSpace_ReInvM, zSpace_ImInvM = 'linArch_N500_L3_relu_tanh_Z5_Re', 'linArch_N1000_L3_relu_tanh_Z5_Im'

    zSpaceDicts = {'ReM': commRoot_Mat + zSpace_ReM + vaeStr,
                   'ImM': commRoot_Mat + zSpace_ImM + vaeStr,
                   'ReInvM': commRoot_Inv + zSpace_ReInvM + vaeStr,
                   'ImInvM': commRoot_Inv + zSpace_ImInvM + vaeStr}
    fitHandle = trainCard['Re//Im'] + 'InvM'

    for zSpaceType in zSpaceDicts.keys():
        with open(zSpaceDicts[zSpaceType], 'rb') as pZIn:
            zSpace_Dict = pickle.load(pZIn)
        zSpaceDicts[zSpaceType] = zSpace_Dict

    # Set the random seed and load the data
    rndSeed = 0
    np.random.seed(rndSeed)

    # NN declarations
    trainArray_Re, trainArray_Im = zSpaceDicts['ReM']['Zspace'], zSpaceDicts['ImM']['Zspace']

    # Normalise and concatenate
    reNormer = ArrayNormaliser(trainArray_Re, trainCard['normData'])
    imNormer = ArrayNormaliser(trainArray_Im, trainCard['normData'])

    trainSpaceDict_Re = reNormer.normData(trainArray_Re)
    trainArray_Re = trainSpaceDict_Re['NormArray']

    trainSpaceDict_Im = imNormer.normData(trainArray_Im)
    trainArray_Im = trainSpaceDict_Im['NormArray']

    trainSpace = np.concatenate((flattenArray(trainArray_Re), flattenArray(trainArray_Im)), axis=1)

    # Fit with the neural networks
    inputShape = trainSpace.shape[1]
    fitFct = 'sigmoid'
    nbNeur = trainCard['nbOfNeurons']
    mBatch = trainCard['mBatch']
    nbEpochs = trainCard['nbOfEpochs']
    nbLayers = trainCard['nbLayers']
    zDim = 5

    fitModel = makeLinArch(inputShape, nbNeur, nbLayers + 1, zDim)

    # Fit and save the Z space mappings and the normalisation instance
    from halo import Halo
    spinner = Halo(text=f'Training hard for the money so hard for the moneyehhh 💵💵💵', spinner='moon')
    spinner.start()

    earlyStopping = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)]
    fitHistory = fitModel.fit(trainSpace, zSpaceDicts[fitHandle]['Zspace'], batch_size=mBatch,
                              epochs=nbEpochs, validation_split=0.2, verbose=1, callbacks=earlyStopping)
    spinner.stop_and_persist(symbol='🌞', text='It done gud.')
    showStats(fitHistory)

    # Save the result of the fit
    zStr = 'zSpaceMaps/latZfit_10-5'
    fitModel.save(zStr + '_' + trainCard['Re//Im'])
    with open(zStr + '_' + trainCard['Re//Im'] + '/Znorm.p', 'wb') as pcklOut:
        pickle.dump({"Re": reNormer, "Im": imNormer}, pcklOut)
