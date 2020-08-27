import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import warnings

for insertPath in ['../', '../examples', 'Utils']:
    sys.path.insert(1, insertPath)

# from normArrays import ArrayNormaliser
# from latmattools.manip import manip
# np.set_printoptions(precision=2, threshold=sys.maxsize)

#  FNULL declaration to supress cmd line output ####
import os
import subprocess
FNULL = open(os.devnull, 'w')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


class TerminateOnBaseline(Callback):
    '''
        Callback that terminates training when either acc or val_acc reaches a specified baseline
    '''

    def __init__(self, monitor='total_loss', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.epochLosses = []

    def on_batch_end(self, mBatchNb, logs=None):
        # print(f'On batch number {mBatchNb} have a loss of {logs.get(self.monitor)}')
        self.epochLosses.append(logs.get(self.monitor))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = np.mean(self.epochLosses)
        # print(f'\nMonitoring {self.monitor} with value {acc:.2f}, on end of epoch {epoch}.')
        self.epochLosses = []

        if acc is not None:
            if acc <= self.baseline:
                print('\nEpoch %d: Reached baseline, terminating training' % (epoch))
                print(acc, self.baseline)
                self.model.stop_training = True


#  Sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            # Account for the dimension of the data???
            reconstruction_loss *= data.shape[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        lossDict = {"total_loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
        return lossDict


def plotZspace(encoder, data, figHandle, labels='IndxPos', runDict={}):
    '''
        Given uncompressed high dimensional data the function uses the encoder prediction for the Z space, which is
        then plotted in a 2D array. It is then saved into a pickle file in the latent Z space directory.
    '''
    nbPoints = data.shape[0]
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))

    saveHandle = figHandle + '_' + runDict['VAE'] + '/'
    subprocess.call('mkdir LatentZspaces/' + saveHandle, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    # Use the matrix number as the color ID to correlate between matrices and their inverse
    if labels == 'IndxPos':
        labels = np.array([i for i in range(nbPoints)])
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig('LatentZspaces/' + saveHandle + 'LatZ.pdf')
    plt.show()

    # Export the Z space with the run dictionary
    pckDict = {'runDict': runDict, 'Zspace': z_mean}
    with open('LatentZspaces/' + saveHandle + 'VAE_Info.p', 'wb') as pcklOut:
        pickle.dump(pckDict, pcklOut)


def mkLayerDict(n_dimm, nbNeur, nbLayers, layerAct, dropRate):
    '''
        Makes a layer dictionary including dropout layers.
    '''

    encoder_inputs = keras.Input(shape=(n_dimm,))

    layerDict = {}
    dropDict = {}
    for layerNb in range(nbLayers):
        layerDict[str(layerNb)] = layers.Dense(nbNeur, activation=layerAct)
        dropDict[str(layerNb)] = layers.Dropout(dropRate)

    layerDict['0'] = layerDict['0'](encoder_inputs)
    dropDict['0'] = dropDict['0'](layerDict['0'])
    for layerNb in range(nbLayers - 1):
        # layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](layerDict[str(layerNb)])
        layerDict[str(layerNb + 1)] = layerDict[str(layerNb + 1)](dropDict[str(layerNb)])
        dropDict[str(layerNb + 1)] = dropDict[str(layerNb + 1)](layerDict[str(layerNb)])

    return layerDict, dropDict, encoder_inputs


def mkSaveDirs(auxHandle, nbNeur, nbLayers, layerAct, outNeuron, latent_dim):
    '''
        Makes the required save ditectory and returns the VAE generated Id.
    '''
    # Save models
    vaeID = f'{auxHandle[0]}-linArch_N{nbNeur}_L{nbLayers}_{layerAct}_{outNeuron}_Z{latent_dim}_{auxHandle[1]}'
    subprocess.call('mkdir VAEs/', shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    dirToMake = 'VAEs/' + vaeID + '/'
    subprocess.call('mkdir ' + dirToMake, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    subDirs = ['Encoder/', 'Decoder/']
    for subDir in subDirs:
        subprocess.call('mkdir ' + dirToMake + subDir, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    return vaeID, dirToMake


def linearArch(matArray, nbNeur=500, nbLayers=1, outNeuron='tanh', layerAct='relu', latent_dim=2, dropRate=0.8,
               auxHandle=None):
    '''
        Returns the linear architecture VAE.
    '''
    nbMatrices = matArray.shape[0]
    trainMats = []
    for i in range(nbMatrices):
        trainMats.append(matArray[i].flatten())
    trainMats = np.array(trainMats)
    n_dimm = trainMats.shape[1]

    # Save models
    vaeID, dirToMake = mkSaveDirs(auxHandle, nbNeur, nbLayers, layerAct, outNeuron, latent_dim)

    # Rectified Linear Unit architecture
    layerDict, dropDict, encoder_inputs = mkLayerDict(n_dimm, nbNeur, nbLayers, layerAct, dropRate)

    # Build latent space
    z_mean = layers.Dense(latent_dim, name="z_mean")(dropDict[str(nbLayers - 1)])
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(dropDict[str(nbLayers - 1)])
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    #  Build decoder
    layerDict_dec, dropDict_dec, latent_inputs = mkLayerDict(latent_dim, nbNeur, nbLayers, layerAct, dropRate)

    decoder_outputs = layers.Dense(n_dimm, activation=outNeuron)(dropDict_dec[str(nbLayers - 1)])
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    if True:
        keras.utils.plot_model(encoder, dirToMake + "Encoder/encoderModel.png", show_shapes=True)
        keras.utils.plot_model(decoder, dirToMake + "Decoder/decoderModel.png", show_shapes=True)
        encoder.save(dirToMake + "Encoder/")
        decoder.save(dirToMake + "Decoder/")

    return encoder, decoder, trainMats, vaeID


def loadAllMats():
    '''
        Load all the matrices from the pickle jars.
    '''
    matArray = pickle.load(open('../examples/picklejar/allmats-10000', 'rb'))
    invMatArray = pickle.load(open('../examples/picklejar/allinvmats-10000', 'rb'))

    return {'matArray': matArray, 'invMatArray': invMatArray}


def splitMatArrs(matArray, trainSplit):
    '''
        Split the matrix arrays accordnig to the trainSplit fraction.
    '''
    # Set the random seed and load the data
    rndSeed = 0
    np.random.seed(rndSeed)
    boolMask = np.array([True if np.random.uniform() < trainSplit else False
                         for _ in range(matArray.shape[0])
                         ])
    return matArray[boolMask], matArray[np.logical_not(boolMask)]


def splitReIm(matArray):
    '''
        Given a matrix array the funciton returns two arrays, one with the real parts and the other with
        imaginary parts.
    '''
    return matArray.real, matArray.imag


def mkDir(dirStr):
    '''
        Make a directory at the location specified in dirStr.
    '''
    subDirs = dirStr.split('/')

    dirStruct = ''
    for subDir in subDirs:
        dirStruct = dirStruct + subDir + '/'
        subprocess.call('mkdir ' + dirStruct, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)


def showStats(history):
    '''
        Show statistics related to the fitting.
    '''
    saveDir = 'AnalysisPlots/VAE_losses/'
    mkDir(saveDir)

    # Show the training losses
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(history.history['total_loss'], c='C0')
    ax1.set_title('VAE loss.')

    ax2.plot(history.history['reconstruction_loss'], c='C1')
    ax2.set_title('RecLoss.')

    ax3.plot(history.history['kl_loss'], c='C2')
    ax3.set_title('KL loss.')

    plt.savefig(saveDir + 'trainLosses.pdf')
    plt.show()


def plotZspace(encoder, data, figHandle, labels='IndxPos', runDict={}):
    '''
        Given uncompressed high dimensional data the function uses the encoder prediction for the Z space, which is
        then plotted in a 2D array. It is then saved into a pickle file in the latent Z space directory.
    '''
    nbPoints = data.shape[0]
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))

    saveHandle = figHandle + '_' + runDict['VAE'] + '/'
    mkDir('LatentZspaces/' + saveHandle)

    # Use the matrix number as the color ID to correlate between matrices and their inverse
    if labels == 'IndxPos':
        labels = np.array([i for i in range(nbPoints)])
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig('LatentZspaces/' + saveHandle + 'LatZ.pdf')
    plt.show()

    # Export the Z space with the run dictionary
    pckDict = {'runDict': runDict, 'Zspace': z_mean}
    with open('LatentZspaces/' + saveHandle + 'VAE_Info.p', 'wb') as pcklOut:
        pickle.dump(pckDict, pcklOut)


def initParser():
    '''
        Initialise and parse the arguments.
    '''
    parser = argparse.ArgumentParser(description='Process the inputs for the plotting function')
    parser.add_argument('mat//invMat', help='Input mat to train the matrix array, invMat for the inverse.')
    parser.add_argument('Re//Im', help='Input Re to train the real part, Im to train the imaginary part')

    parser.add_argument("-nE", '--nbOfEpochs', help='Specify number of training epochs.',
                        type=int, default=10000)
    parser.add_argument("-mB", '--mBatch', help='Specify minibatch size.',
                        type=int, default=128)
    parser.add_argument("-nNN", '--nbOfNeurons', help='Specify number of neurons for flat architecture with N layers.', type=int, default=250)
    parser.add_argument("-nL", '--nbOfLayers', help='Specify number of Layers.', type=int, default=1)
    parser.add_argument('--normData', help='Set flag to normalise data as either UnitVec or SetMeanStd', type=str, default='')
    parser.add_argument('--trainSplit', help='Specify the training split, default to 0.8.', type=float, default=0.85)
    parser.add_argument('--baseKill', help='Specify the total loss value at which the training stops.', type=float, default=1.0)
    argsPars = parser.parse_args()
    trainCard = vars(argsPars)

    return trainCard


if __name__ == '__main__':
    trainCard = initParser()
    # Load params
    runSet = trainCard['mat//invMat'] + 'Array'
    runPart = trainCard['Re//Im']
    trainSplit = trainCard['trainSplit']

    # Set the random seed and load the data
    rndSeed = 0
    np.random.seed(rndSeed)

    # Load and split the matrices via train/test and Re/Im
    matDict = loadAllMats()
    trainArray, testArray = splitMatArrs(matDict, runSet, trainSplit)
    trainArray_Re, trainArray_Im = splitReIm(trainArray)
    trainDict = {'Re': trainArray_Re, 'Im': trainArray_Im}

    #   Initialise the VAE parameters
    nbEpochs = trainCard['nbOfEpochs']
    mBatch = trainCard['mBatch']
    nbNeur = trainCard['nbOfNeurons']
    nbLayers = trainCard['nbOfLayers']

    # Set out normalisations
    if trainCard['normData'] == 'UnitVec' and trainCard['mat//invMat'] == 'invMat':
        raise KeyError('Cannot use unit vector normalisation for the inverse matrix VAE!')
    else:
        matNormer = ArrayNormaliser(trainDict[runPart], trainCard['normData'])
        trainArray = matNormer.normData(trainDict[trainCard['Re//Im']])

    encoder, decoder, trainMats, vaeID = linearArch(trainArray['NormArray'], nbNeur=nbNeur, latent_dim=10, nbLayers=nbLayers+1, auxHandle=[runSet, runPart])

    #  Compile and train VAEs
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(),
                metrics=["total_loss", "reconstruction_loss", "kl_loss"])
    print(trainMats.shape)

    earlyStopping = [
                    EarlyStopping(monitor='total_loss', patience=500, verbose=1, mode='min'),
                    TerminateOnBaseline(monitor='total_loss', baseline=trainCard['baseKill'])
                    ]
    vaeHist = vae.fit(trainMats, epochs=nbEpochs, batch_size=mBatch, verbose=1, callbacks=earlyStopping)

    encoder.save('VAEs/' + vaeID + '/' + 'Encoder/')
    decoder.save('VAEs/' + vaeID + '/' + 'Decoder/')

    showStats(vaeHist)
    # Plot and save the Z space for the training set
    runDict = {'Epochs': nbEpochs, 'mBatch': mBatch, 'VAE': vaeID,
               'Train': runSet, 'TrainSplit': trainSplit, 'RndSeed': rndSeed,
               'FitLosses': vaeHist.history,
               'NormType': trainCard['normData'], 'NormInstance': matNormer
               }
    plotZspace(encoder, trainMats, runSet + '_Zspace', runDict=runDict)
