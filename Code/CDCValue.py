import pandas as pd
import numpy as np
import scipy
import glob
from copy import deepcopy

import TPfeature

class cCDCvalueHandler():
    def __init__(self,
                 gchCinScale = None,
                 gchHoriWidth = None,
                 gchVertWidth = None):
        '''
            Arguments:
                + gchCinScale: Cin Scale 
                + gchHoriWidth: Number of horizontal sensor stage
                + gchVertWidth: Number of vertical sensor stage
        '''
        self.Path = None    # Path of forder or files 
        self.Files = None   # List of files which be wanted to open
        self.cdcValues = [] # Storing the CDC values which read from files
        self.min_cdcValue = None # Minimum CDC value from files 
        self.max_cdcValue = None # Maximum CDC value from files
        
        self.numVertStage = gchVertWidth # Number of vertical sensor stage
        self.numHoriStage = gchHoriWidth # Number of horizontal sensor stage
        self.cinScale = gchCinScale # Cin Scale 
        
        self.cdcSample = None # Sample CDC value from simulation topography or reading from files
        self.cdcSamplePrev = None
        self.cdcSampleDer = None
        self.cdcSampleSpaGraX = None 
        self.cdcSampleSpaGraY = None
        self.PreTopo = None # Pressure topography which generated from simulation or interpolated by CDC value
        self.PreTopoInt = None # Pressure topography which interpolated by CDC value (self.cdcSample)
                               # either the real data or sampling from simulation, the difference between the
                               # self.PreTopo and self.PreTopoInt as following statements:
                               # self.PreTopo:
                               #             1. Generating by SimulationDataGenerated() which uses the
                               #                spatial Gaussian model to simulate the pressure distribution
                               #             2. Using CDCtoTopography() which uses the interpolating method 
                               #                to get the pressure distribution from the real CDC sample
                               # self.PreTopoInt:
                               #             It is used for simulation when one has generated the simulated
                               #             topography and CDC sample, then one can use the CDC sample to 
                               #             interpolate value of self.PreTopoInt. This case simulates that
                               #             one uses real CDC sample to get the pressure topogaphy. 
        self.Xgrid = None
        self.Ygrid = None
    def ReadValues(self, stPath = None, lStage = None):
        '''
            Arguments:
                + stPath: Path of forder or files
                + lStage: List of stages which one wants to extract
                    Example: ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3',
                              'Stage 4', 'Stage 5', 'Stage 6', 'Stage 7',
                              'Stage 8', 'Stage 9', 'Stage 10', 'Stage 11']
        '''
        tmp_cdcValue = []
        self.Path = stPath
        self.Files = glob.glob(self.Path)
        for fileName in self.Files:
            tmpCDCfile = pd.read_csv(fileName)
            self.cdcValues.append(tmpCDCfile[lStage].values)
        tmp_cdcValue = self.cdcValues[0]
        for i in range(1,len(self.cdcValues)):
            tmp_cdcValue=np.concatenate((tmp_cdcValue, self.cdcValues[i]),axis = 0)
        self.cdcValues = np.uint16(deepcopy(tmp_cdcValue))
        self.min_cdcValue = np.min(self.cdcValues)
        self.max_cdcValue = np.max(self.cdcValues)
    def CDCtoTopography(self, 
                        npaCdcValue = None,
                        iTopo = None,
                        stMethod = 'cubic'):
        '''
            Using the interpolation to get the topography from CDC samples.
            Arguments:
                + cdcValue: CDC samples
                + iTopo: Storing the value of topography
                + stMethod: Method of interpolation. Default is 'cubic'.
                          Future: Using another interpolating methods to compare
        '''                
        # Grid for topography
        if self.numHoriStage is None and self.cinScale is None:
            self.Xgrid = np.linspace(0, 2, 192)
        else:
            self.Xgrid = np.linspace(0,self.numHoriStage-1, (self.numHoriStage-1) * self.cinScale)
        
        if self.numVertStage is None and self.cinScale is None:
            self.Ygrid = np.linspace(0,3,264)
        else:
            self.Ygrid = np.linspace(0,self.numVertStage-1, (self.numVertStage-1) * self.cinScale)

        self.Xgrid, self.Ygrid = np.meshgrid(self.Xgrid, self.Ygrid)

        # Grid for CDC sample
        if self.numHoriStage is None:
            x = np.linspace(0,2,3)
        else:
            x = np.linspace(0,self.numHoriStage-1,self.numHoriStage)

        if self.numVertStage is None and self.cinScale is None:
            y = np.linspace(0,3,4)
        else:
            y = np.linspace(0,self.numVertStage-1, self.numVertStage)

        grid_x, grid_y = np.meshgrid(x,y)

        # Grid coordinate for CDC sample 
        gridCoord = [grid_y.flatten(),grid_x.flatten()]
        gridCoord = np.asarray(gridCoord).T

        if iTopo == 0:
            self.PreTopo = griddata(gridCoord, npaCdcValue.flatten(), (self.Ygrid, self.Xgrid), method = stMethod)
        elif iTopo == 1:
            self.PreTopoInt = griddata(gridCoord, npaCdcValue.flatten(), (self.Ygrid, self.Xgrid), method = stMethod)
    def SimulationDataGenerated(self, 
                                npaMu = None, 
                                npaSigma = None,
                                iPreInt = None,
                                NoiseP = None):
        '''
            Using the spatial Gaussian model to model the pressure distribution during touch.
            Arguments:
                + npaMu: Mean of Gaussian distribution
                + npaSigma: Sigma of Gaussian distribution
                + iPreInt: Pressure intensity
        '''
        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos.
        
            pos is an array constructed by packing the meshed arrays of variables
            x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        
            """
        
            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2*np.pi)**n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
            return np.exp(-1* fac / 2) / N

        # Grid for topography
        if self.numHoriStage is None and self.cinScale is None:
            self.Xgrid = np.linspace(0, 2, 192)
        else:
            self.Xgrid = np.linspace(0,self.numHoriStage, (self.numHoriStage) * self.cinScale)
        
        if self.numVertStage is None and self.cinScale is None:
            self.Ygrid = np.linspace(0,3,264)
        else:
            self.Ygrid = np.linspace(0,self.numVertStage, (self.numVertStage) * self.cinScale)

        self.Xgrid, self.Ygrid = np.meshgrid(self.Xgrid, self.Ygrid)
        tmpPos = np.empty(self.Xgrid.shape + (2,))
        tmpPos[:, :, 0] = self.Xgrid
        tmpPos[:, :, 1] = self.Ygrid

        # Generate the pressure distribution
        self.PreTopo = multivariate_gaussian(tmpPos, npaMu, npaSigma)*iPreInt
        stride = self.cinScale
        
        # Store the previous CDC sample
        if self.cdcSample is not None:
            self.cdcSamplePrev = self.cdcSample

        '''
        Sample the CDC value from pressure distribution
        If there doesn't exist the noise parameter NoiseP,
        this means that the system free from noise. 
        NoiseP is used to generate the Gaussian white noise
        which Sigma is NoiseP[0] and Mu is NoiseP[1].
        CDC samples from prssure topography self.PreTopo by
        (0,0) : (stride//2-1, stride//2-1)
        (0,1) : (stride//2-1, stride//2 + stride -1)
        ....
        ''' 
        if NoiseP is None:
            self.cdcSample = np.uint32(np.asarray([[self.PreTopo[stride//2-1, stride//2-1], 
                                                    self.PreTopo[stride//2-1, stride//2+stride-1],
                                                    self.PreTopo[stride//2-1, stride//2+stride*2-1]],
                                                [self.PreTopo[stride//2+stride-1, stride//2-1], 
                                                    self.PreTopo[stride//2+stride-1, stride//2+stride-1],
                                                    self.PreTopo[stride//2+stride-1, stride//2+stride*2-1]],
                                                [self.PreTopo[stride//2+stride*2-1, stride//2-1], 
                                                    self.PreTopo[stride//2+stride*2-1, stride//2+stride-1],
                                                    self.PreTopo[stride//2+stride*2-1, stride//2+stride*2-1]],
                                                [self.PreTopo[stride//2+stride*3-1, stride//2-1], 
                                                    self.PreTopo[stride//2+stride*3-1, stride//2+stride-1],
                                                    self.PreTopo[stride//2+stride*3-1, stride//2+stride*2-1]]]))
        else:
            '''
                The sensor noise is generated by Gaussian process: Sigma*N(0,1)+Mu
            '''
            sensorNoise = NoiseP[0]*np.random.randn(4,3)+NoiseP[1]
            out = np.asarray([[self.PreTopo[stride//2-1, stride//2-1], 
                               self.PreTopo[stride//2-1, stride//2+stride-1],
                               self.PreTopo[stride//2-1, stride//2+stride*2-1]],
                              [self.PreTopo[stride//2+stride-1, stride//2-1], 
                               self.PreTopo[stride//2+stride-1, stride//2+stride-1],
                               self.PreTopo[stride//2+stride-1, stride//2+stride*2-1]],
                              [self.PreTopo[stride//2+stride*2-1, stride//2-1], 
                               self.PreTopo[stride//2+stride*2-1, stride//2+stride-1],
                               self.PreTopo[stride//2+stride*2-1, stride//2+stride*2-1]],
                              [self.PreTopo[stride//2+stride*3-1, stride//2-1], 
                               self.PreTopo[stride//2+stride*3-1, stride//2+stride-1],
                               self.PreTopo[stride//2+stride*3-1, stride//2+stride*2-1]]])+sensorNoise
            '''
                The negative CDC sample is clipped to 0.
            '''
            self.cdcSample = np.uint32(out.clip(min = 0))
        self.ZeroPadding()
    def GetSpatialGradient(self, cdcValues = None):
        '''
            Delta I(x) and Delta I(y)
            The function caculates the spatial gradient. If the cdcValue is None,
            the spatial gradient caculates the CDC sample from the pressure topography.
            If the cdcValue is provided, this scenario is applied in the real CDC values case.
            + cdcValues: numpy array with shape (1,12)
        '''
        xKernel = np.array([-1,1])
        yKernel = np.array([[-1],[1]])
        if cdcValues is None:
            self.ZeroPadding()
            tmp = np.int32(self.cdcSample)
            self.cdcSampleSpaGraX = np.int32(np.zeros_like(self.cdcSample))                               
            for i in range(1, self.numVertStage+1, 1):
                for j in range(1, self.numHoriStage+1, 1):
                    mask = np.array([np.int32(tmp[i,j-1]), np.int32(tmp[i,j])])
                    self.cdcSampleSpaGraX[i,j] = np.dot(mask,xKernel)
            self.cdcSampleSpaGraY = np.int32(np.zeros_like(self.cdcSample))
            for i in range(1, self.numVertStage+1, 1):
                for j in range(1, self.numHoriStage+1, 1):
                    mask = np.array([ [np.int32(tmp[i-1,j])], [np.int32(tmp[i,j])] ])
                    self.cdcSampleSpaGraY[i,j] = np.dot(mask.flatten(),yKernel.flatten())

            # print('CDC sample :\n{0}'.format(self.cdcSample))
            # print('CDC spatial gradient X axis :\n{0}'.format(self.cdcSampleSpaGraX))
            # print('CDC spatial gradient Y axis :\n{0}'.format(self.cdcSampleSpaGraY))
    def GetTimeDerivative(self, cdcValues = None):
        '''
            Delta I(t)
            The function caculates the time derivative of CDC. If the cdcValues is None,
            the function takes derivation about the CDC sample from the pressure topography.
            If the cdcValue is provided, this scenario is applied in the real CDC values case.
            + cdcValues: numpy array with shape (1,12)
        '''
        if cdcValues is None:
            if self.cdcSamplePrev is not None:
                self.cdcSampleDer = np.int32(self.cdcSample) - np.int32(self.cdcSamplePrev)
    def SaveValueToNpy(self, fileName = None):
        np.save(fileName ,self.cdcValues)
    def ConvertSimCDCtoRealCDCFormat(self):
        return np.array([self.cdcSample[4,1], self.cdcSample[3,1], self.cdcSample[2,1], self.cdcSample[1,1],
                         self.cdcSample[4,2], self.cdcSample[3,2], self.cdcSample[2,2], self.cdcSample[1,2],
                         self.cdcSample[4,3], self.cdcSample[3,3], self.cdcSample[2,3], self.cdcSample[1,3]])
#        return np.array([self.cdcSample[1,1], self.cdcSample[2,1], self.cdcSample[3,1], self.cdcSample[4,1],
#                         self.cdcSample[1,2], self.cdcSample[2,2], self.cdcSample[3,2], self.cdcSample[4,2],
#                         self.cdcSample[1,3], self.cdcSample[2,3], self.cdcSample[3,3], self.cdcSample[4,3]])
    def LoadValueFromNpy(self, fileName = None):
        self.cdcValues = np.load(fileName)
        self.cdcValues = np.uint16(self.cdcValues)
        self.min_cdcValue = np.min(self.cdcValues)
        self.max_cdcValue = np.max(self.cdcValues)
    def ZeroPadding(self, cdcValues = None):
        '''
            Zero padding
        '''
        if cdcValues is None:
            self.cdcSample = np.pad(self.cdcSample, (1,1), 'constant', constant_values = (0))
            # self.cdcSample = np.unit32(np.array([[    0,                   0,                   0,                   0,    0],
            #                                      [    0, self.cdcSample[0,0], self.cdcSample[0,1], self.cdcSample[0,2],    0],
            #                                      [    0, self.cdcSample[1,0], self.cdcSample[1,1], self.cdcSample[1,2],    0],
            #                                      [    0, self.cdcSample[2,0], self.cdcSample[2,1], self.cdcSample[2,2],    0],
            #                                      [    0,                   0,                   0,                   0,    0]]))
    def GetCDCElectrodes(self):
        return self.cdcSample[1:self.numVertStage, 1:self.numHoriStage]
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata

    Method = 'cubic'

    Handler = cCDCvalueHandler(gchVertWidth = 4, 
                               gchHoriWidth = 3,
                               gchCinScale = 64)
    TP = TPfeature.TPfeature(gchVertWidth = 4, 
                             gchHoriWidth = 3,
                             gchCinScale = 64)
#    Handler.ReadValues(stPath = './RealData/*.csv',
#                       lStage = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3',
#                                 'Stage 4', 'Stage 5', 'Stage 6', 'Stage 7',
#                                 'Stage 8', 'Stage 9', 'Stage 10', 'Stage 11'])
#    Handler.SaveValueToNpy(fileName = 'cdcValue')
    
    
#    TestSample = Handler.cdcValues[200,:].reshape((3,4)).T
#    Handler.CDCtoTopography(npaCdcValue = TestSample,
#                            iTopo = 0,
#                            stMethod='cubic')



    Mu = np.array([1.5, 2.2])
    Sigma = np.array([[1,0],
                      [0,1]])
    
    PressureIntensity = 40000
        
    Simulator = cCDCvalueHandler(gchCinScale = 64,
                                 gchHoriWidth = 3,
                                 gchVertWidth = 4)
    Simulator.SimulationDataGenerated(npaMu = Mu,
                                      npaSigma = Sigma,
                                      iPreInt = PressureIntensity)
#    Simulator.CDCtoTopography(npaCdcValue = Simulator.cdcSample,
#                              iTopo = 1,
#                              stMethod = Method)
#    
#    maxIndx = np.unravel_index(np.argmax(Simulator.cdcSample, axis=None), Simulator.cdcSample.shape)
    
#    # Center of mass
#    ySum = (Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]]+Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]]+Simulator.cdcSample[maxIndx])
#    yRatioM = (Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]]-Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]])/ySum 
#    xSum = (Simulator.cdcSample[maxIndx[0],maxIndx[1]-1]+Simulator.cdcSample[maxIndx[0],maxIndx[1]+1]+Simulator.cdcSample[maxIndx])
#    xRatioM = (Simulator.cdcSample[maxIndx[0],maxIndx[1]+1]-Simulator.cdcSample[maxIndx[0],maxIndx[1]-1])/xSum 
#    
#    # Linear 
#    yMin = np.min([Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]],Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]]])
#    yDen = Simulator.cdcSample[maxIndx] - yMin
#    yNeu = Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]]-Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]]
#    yRatioL = yNeu/yDen  * 0.5
#    
#    xMin = np.min([Simulator.cdcSample[maxIndx[0],maxIndx[1]+1],Simulator.cdcSample[maxIndx[0],maxIndx[1]-1]])
#    xDen = Simulator.cdcSample[maxIndx] - xMin
#    xNeu = Simulator.cdcSample[maxIndx[0],maxIndx[1]+1]-Simulator.cdcSample[maxIndx[0],maxIndx[1]-1]
#    xRatioL = xNeu/xDen * 0.5
#    
#    # Parabolic
#    yDen = Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]]+Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]] - 2*Simulator.cdcSample[maxIndx]
#    yNeu = Simulator.cdcSample[maxIndx[0]+1,maxIndx[1]]-Simulator.cdcSample[maxIndx[0]-1,maxIndx[1]]
#    yRatioP = yNeu/yDen  * 0.5
#    
#    xDen = Simulator.cdcSample[maxIndx[0],maxIndx[1]+1]+Simulator.cdcSample[maxIndx[0],maxIndx[1]-1] - 2*Simulator.cdcSample[maxIndx]
#    xNeu = Simulator.cdcSample[maxIndx[0],maxIndx[1]+1]-Simulator.cdcSample[maxIndx[0],maxIndx[1]-1]
#    xRatioP = xNeu/xDen * 0.5
    
    
    
    # fig = plt.figure()
    # plt.subplot(1,2,1)
    # psm = plt.pcolormesh(TestSample, cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
    # plt.title('Real CDC data')
    # plt.subplot(1,2,2)
    # psm = plt.pcolormesh(np.real(Handler.PreTopo), cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
    # plt.title('CDC to tomography (CUBIC)')


#    fig = plt.figure()
#    st = fig.suptitle(Method, fontsize="x-large")
#    ax1 = plt.subplot(2,2,1)
#    psm = plt.pcolormesh(np.real(Simulator.PreTopo), cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
#    ax1.set_title('Simulation pressure tomography')
#    ax2 = plt.subplot(2,2,2)
#    psm = plt.pcolormesh(np.real(Simulator.cdcSample), cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
#    ax2.set_title('Simulation CDC value')
#    ax3 = plt.subplot(2,2,3)
#    psm = plt.pcolormesh(np.real(Simulator.PreTopoInt), cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
#    ax3.set_title('Interpolation of pressure distribution')
#    ax4 = plt.subplot(2,2,4)
#    psm = plt.pcolormesh(np.real(Simulator.PreTopo)-np.real(Simulator.PreTopoInt), cmap = 'plasma',vmin = Handler.min_cdcValue, vmax = Handler.max_cdcValue)
#    ax4.set_title('Residue')
#    plt.show()

       









    
    
    
    
    

    
    
    