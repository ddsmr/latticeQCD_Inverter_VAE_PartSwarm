import numpy as np
from time import gmtime, strftime

# Pretty python traceback outputs
try:
    import colored_traceback
    colored_traceback.add_hook()
except Exception as e:
    print(e)


class ArrayNormaliser:
    '''
        Normalisation class, has the direct and the inverse transformations included as methods.
    '''

    def __init__(self, datArray, normType):
        '''
            Initialise the normalisation class with the array values and the normalisation type. Save the required
            attributes in the normalisation dictionary.

            If normType is:
                - 'SetMeanStd': normalise data via Z = (X - μ) / σ.
                - 'UnitVec': normalise each matrix as M / det(M).
        '''
        self.normType = normType
        self.normDict = {}
        self.setID = ''
        if normType == 'SetMeanStd':
            # self.normDict['Norm'] = {'Mean': np.mean(datArray), 'Std': np.std(datArray)}
            self.normDict['Norm'] = {'Mean': np.ndarray.mean(datArray, axis=0),
                                     'Std': np.std(datArray)}
    def normData(self, datArray):
        '''
            Normalise the data via the direct transformation.
        '''
        if self.normType == '':
            return {'NormArray': datArray, 'SetID': ''}
        elif self.normType == 'SetMeanStd':
            normArray = (datArray - self.normDict['Norm']['Mean']) / self.normDict['Norm']['Std']
            return {'NormArray': normArray, 'SetID': ''}

        elif self.normType == 'UnitVec':
            currTime = strftime("-%d%m%Y%H%M%S", gmtime())
            self.setID = 'DataSet-' + str(datArray.shape) + '-' + str(int(np.random.uniform(1, 1000))) + currTime
            nbMatrices = datArray.shape[0]
            normArray = []

            for matNb in range(nbMatrices):
                unitNorm = np.linalg.det(datArray[matNb])
                self.normDict[str(matNb)] = unitNorm
                normArray.append(datArray[matNb] / unitNorm)

            return {'NormArray': np.array(normArray), 'SetID': self.setID}

    def invNormData(self, invDatArray, setID):
        '''
            Perform the inverse normalisation on the inverse data array.
        '''
        if self.normType == '':
            return invDatArray
        elif self.normType == 'SetMeanStd':
            deNormArray = invDatArray * self.normDict['Norm']['Std'] + self.normDict['Norm']['Mean']
            return deNormArray
        elif self.normType == 'UnitVec':
            if setID != self.setID:
                raise KeyError('The inverse data set does not correspond to the original set.')
            else:
                nbMatrices = invDatArray.shape[0]
                deNormArray = []
                for matNb in range(nbMatrices):
                    unitNorm = self.normDict[str(matNb)]
                    deNormArray.append(invDatArray[matNb] * unitNorm)
                print(len(deNormArray))
                return np.array(deNormArray)


if __name__ == '__main__':
    '''
       Test the functionality of the normalisation class.
    '''
    xArrTest = np.random.uniform(size=(100, 5, 5))
    matNormer = ArrayNormaliser(xArrTest, 'UnitVec')
    normDict = matNormer.normData(xArrTest)

    denormData = matNormer.invNormData(normDict['NormArray'], normDict['SetID'])
    print(xArrTest[0], '\n\n', normDict['NormArray'][0], '\n\n', denormData[0])
