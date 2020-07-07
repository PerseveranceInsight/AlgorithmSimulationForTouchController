import numpy as np
import CDCValue

class cCoordTransfor():
    '''
            Arguments:
                + gchCinScale: Cin Scale 
                + gchHoriWidth: Number of horizontal sensor stage
                + gchVertWidth: Number of vertical sensor stage
    '''
    def __init__(self,
                 gchCinScale = None,
                 gchHoriWidth = None,
                 gchVertWidth = None):
        self.lTrace = [] #Storing the trace of touching

        self.gchCinScale = gchCinScale
        self.gchHoriWidth = gchHoriWidth
        self.gchVertWidth = gchVertWidth

        '''
            This generates the coordinates of simulated CDC electrodes. If one
            uses the real CDC values which needs to coordinate transform.
            (0,0) : 0: (32,32)
            (0,1) : 1: (32,96)
            (0,2) : 2: (32,160)
            (1,0) : 3: (96,32)
        '''  
        self.Xgrid = np.linspace(0, self.gchHoriWidth-1, self.gchHoriWidth)
        self.Ygrid = np.linspace(0, self.gchVertWidth-1, self.gchVertWidth)
        self.Xgrid, self.Ygrid = np.meshgrid(self.Xgrid, self.Ygrid)
        self.Xgrid = self.Xgrid*self.gchCinScale+32
        self.Ygrid = self.Ygrid*self.gchCinScale+32

        '''
            Dictionary for coordinate transformation
            The keys of this dictionary are the index of CDC sample from simulation. 
            Real CDC electrodes are value of this dictionary.

                Order of simulation(Order of real data)
                0(3)  1(7)  2(11)
                3(2)  4(6)  5(10)
                6(1)  7(5)  8( 9)
                9(0) 10(4) 11( 8)
        '''
        self.dIndTranSimToReal = {0:3,
                                  1:7,
                                  2:11,
                                  3:2,
                                  4:6,
                                  5:10,
                                  6:1,
                                  7:5,
                                  8:9,
                                  9:0,
                                  10:4,
                                  11:8}
        self.dIndTranRealToSim = {3:0,
                                  7:1,
                                  11:2,
                                  2:3,
                                  6:4,
                                  10:5,
                                  1:6,
                                  5:7,
                                  9:8,
                                  0:9,
                                  4:10,
                                  8:11}
        '''
            2D position of simulated electrodes
        '''
        self.SimuPos = {0:  (0,0),
                        1:  (0,1),
                        2:  (0,2),
                        3:  (1,0),
                        4:  (1,1),
                        5:  (1,2),
                        6:  (2,0),
                        7:  (2,1),
                        8:  (2,2),
                        9:  (3,0),
                        10: (3,1),
                        11: (3,2)}
        self.SimuPosToInd = {pos: keys for keys, pos in self.SimuPos.items()}
        self.RealPos = {3 : np.array([   0,   0]),
                        7 : np.array([  96,   0]),
                        11: np.array([ 192,   0]),
                        2 : np.array([   0,  85]),
                        6 : np.array([  96,  85]),
                        10: np.array([ 192,  85]),
                        1 : np.array([   0, 170]),
                        5 : np.array([  96, 170]),
                        9 : np.array([ 192, 170]),
                        0 : np.array([   0, 256]),
                        4 : np.array([  96, 256]),
                        8 : np.array([ 192, 256]),}

        '''
            Dictionary for storing the spatial average mask.
            The keys of this dictionary are the index of CDC sample from simulation. 
            The values of this dictionary are the mask coefficients of corresponding index.
        '''
        self.dLow = {0 : np.array([1,1,0,1,1,0,0,0,0,0,0,0]),
                     1 : np.array([1,1,1,1,1,1,0,0,0,0,0,0]),
                     2 : np.array([0,1,1,0,1,1,0,0,0,0,0,0]),
                     3 : np.array([1,1,0,1,1,0,1,1,0,0,0,0]),
                     4 : np.array([1,1,1,1,1,1,1,1,1,0,0,0]),
                     5 : np.array([0,1,1,0,1,1,0,1,1,0,0,0]),
                     6 : np.array([0,0,0,1,1,0,1,1,0,1,1,0]),
                     7 : np.array([0,0,0,1,1,1,1,1,1,1,1,1]),
                     8 : np.array([0,0,0,0,1,1,0,1,1,0,1,1]),
                     9 : np.array([0,0,0,0,0,0,1,1,0,1,1,0]),
                     10: np.array([0,0,0,0,0,0,1,1,1,1,1,1]),
                     11: np.array([0,0,0,0,0,0,0,1,1,0,1,1])}
        '''
            Dictionary for storing the spatial Gaussian mask.
            The keys of this dictionary are the index of CDC sample from simulation. 
            The values of this dictionary are the mask coefficients of corresponding index.
            The mask is 

            1/16 * np.array([[ 1, 2, 1],
                             [ 2, 4, 2],
                             [ 1, 2, 1]])
        '''
        self.dGaussian = {0 : np.array([4,2,0,2,1,0,0,0,0,0,0,0]),
                          1 : np.array([2,4,2,1,2,1,0,0,0,0,0,0]),
                          2 : np.array([0,2,4,0,1,2,0,0,0,0,0,0]),
                          3 : np.array([2,1,0,4,2,0,2,1,0,0,0,0]),
                          4 : np.array([1,2,1,2,4,2,1,2,1,0,0,0]),
                          5 : np.array([0,1,2,0,2,4,0,1,2,0,0,0]),
                          6 : np.array([0,0,0,2,1,0,4,2,0,2,1,0]),
                          7 : np.array([0,0,0,1,2,1,2,4,2,1,2,1]),
                          8 : np.array([0,0,0,0,1,2,0,2,4,0,1,2]),
                          9 : np.array([0,0,0,0,0,0,2,1,0,4,2,0]),
                          10: np.array([0,0,0,0,0,0,1,2,1,2,4,2]),
                          11: np.array([0,0,0,0,0,0,0,1,2,0,2,4])}
        '''
            The real electrodes on the boundary of X axis.
            
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''   
        self.XEdge = [0,1,2,3,8,9,10,11]
        '''
            The real electrodes on the boundary of Y axis.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''   
        self.YEdge = [0,3,4,7,8,11]
        self.Neighbour = {0:  [(0,0), (0,1), (1,0), (1,1)],
                          1:  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
                          2:  [(0,1), (0,2), (1,1), (1,2)],
                          3:  [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],
                          4:  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
                          5:  [(0,1), (0,2), (1,1), (1,2), (2,1), (2,2)],
                          6:  [(1,0), (1,1), (2,0), (2,1), (3,0), (3,1)],
                          7:  [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)],
                          8:  [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)],
                          9:  [(2,0), (2,1), (3,0), (3,1)],
                          10: [(2,0), (2,1), (2,2), (3,0), (3,1), (3,2)],
                          11: [(2,1), (2,2), (3,1), (3,2)]}
        self.MUTURE_THRESHOLD = 100
        
    def remapping(self, xTem, yTem):
        return np.asarray([np.uint32(xTem-32)*(self.gchHoriWidth*self.gchCinScale)/( (self.gchHoriWidth-1)*self.gchCinScale),
                            np.uint32(yTem-32)*(self.gchVertWidth*self.gchCinScale)/( (self.gchVertWidth-1)*self.gchCinScale)])  
    def remappingX(self, xTem):
        return np.uint32(xTem-32)*(self.gchHoriWidth*self.gchCinScale)/( (self.gchHoriWidth-1)*self.gchCinScale)
    def remappingY(self, yTem):
        return np.uint32(yTem-32)*(self.gchVertWidth*self.gchCinScale)/( (self.gchVertWidth-1)*self.gchCinScale)
    def calWeiPos4Adj(self, CDCValue = None):
        '''
            Cacluating the touch position by the 4-adjoint CDC electrodes by centroid.
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        lMaxIndex =  np.int8(np.argmax(CDCValue)) # Find the maximum index of CDCValue.
        if lMaxIndex == 0:
            xDen = np.int32(CDCValue[0]) + np.int32(CDCValue[4])
            xNeu = np.int32(CDCValue[0])*self.Xgrid[self.dIndTranRealToSim[0]] + \
                   np.int32(CDCValue[4])*self.Xgrid[self.dIndTranRealToSim[4]]
            yDen = np.int32(CDCValue[0]) + np.int32(CDCValue[1])
            yNeu = np.int32(CDCValue[0])*self.Ygrid[self.dIndTranRealToSim[0]] + \
                   np.int32(CDCValue[1])*self.Ygrid[self.dIndTranRealToSim[1]]
        elif lMaxIndex == 1:
            xDen = np.int32(CDCValue[1]) + np.int32(CDCValue[5])
            xNeu = np.int32(CDCValue[1])*self.Xgrid[self.dIndTranRealToSim[1]] + \
                   np.int32(CDCValue[5])*self.Xgrid[self.dIndTranRealToSim[5]]
            yDen = np.int32(CDCValue[2]) + np.int32(CDCValue[1]) + np.int32(CDCValue[0])
            yNeu = np.int32(CDCValue[2])*self.Ygrid[self.dIndTranRealToSim[2]] + \
                   np.int32(CDCValue[1])*self.Ygrid[self.dIndTranRealToSim[1]] + \
                   np.int32(CDCValue[0])*self.Ygrid[self.dIndTranRealToSim[0]]
        elif lMaxIndex == 2:
            xDen = np.int32(CDCValue[2]) + np.int32(CDCValue[6])
            xNeu = np.int32(CDCValue[2])*self.Xgrid[self.dIndTranRealToSim[2]] + \
                   np.int32(CDCValue[6])*self.Xgrid[self.dIndTranRealToSim[6]]
            yDen = np.int32(CDCValue[3]) + np.int32(CDCValue[2]) + np.int32(CDCValue[1])
            yNeu = np.int32(CDCValue[1])*self.Ygrid[self.dIndTranRealToSim[1]] + \
                   np.int32(CDCValue[2])*self.Ygrid[self.dIndTranRealToSim[2]] + \
                   np.int32(CDCValue[3])*self.Ygrid[self.dIndTranRealToSim[3]]
        elif lMaxIndex == 3:
            xDen = np.int32(CDCValue[3]) + np.int32(CDCValue[7])
            xNeu = np.int32(CDCValue[3])*self.Xgrid[self.dIndTranRealToSim[3]] +\
                   np.int32(CDCValue[7])*self.Xgrid[self.dIndTranRealToSim[7]]
            yDen = np.int32(CDCValue[3]) + np.int32(CDCValue[2])
            yNeu = np.int32(CDCValue[2])*self.Ygrid[self.dIndTranRealToSim[2]] +\
                   np.int32(CDCValue[3])*self.Ygrid[self.dIndTranRealToSim[3]]
        elif lMaxIndex == 4:
            xDen = np.int32(CDCValue[0]) + np.int32(CDCValue[4]) + np.int32(CDCValue[8])
            xNeu = np.int32(CDCValue[0])*self.Xgrid[self.dIndTranRealToSim[0]] + \
                   np.int32(CDCValue[4])*self.Xgrid[self.dIndTranRealToSim[4]] + \
                   np.int32(CDCValue[8])*self.Xgrid[self.dIndTranRealToSim[8]]
            yDen = np.int32(CDCValue[5]) + np.int32(CDCValue[4])
            yNeu =  np.int32(CDCValue[5])*self.Ygrid[self.dIndTranRealToSim[5]] + \
                    np.int32(CDCValue[4])*self.Ygrid[self.dIndTranRealToSim[4]]
        elif lMaxIndex == 5:
            xDen = np.int32(CDCValue[1]) + np.int32(CDCValue[5]) + np.int32(CDCValue[9])
            xNeu = np.int32(CDCValue[1])*self.Xgrid[self.dIndTranRealToSim[1]] + \
                   np.int32(CDCValue[5])*self.Xgrid[self.dIndTranRealToSim[5]] + \
                   np.int32(CDCValue[9])*self.Xgrid[self.dIndTranRealToSim[9]]
            yDen = np.int32(CDCValue[6]) + np.int32(CDCValue[5]) + np.int32(CDCValue[4])
            yNeu = np.int32(CDCValue[6])*self.Ygrid[self.dIndTranRealToSim[6]] + \
                   np.int32(CDCValue[5])*self.Ygrid[self.dIndTranRealToSim[5]] + \
                   np.int32(CDCValue[4])*self.Ygrid[self.dIndTranRealToSim[4]]
        elif lMaxIndex == 6:
            xDen = np.int32(CDCValue[2]) + np.int32(CDCValue[6]) + np.int32(CDCValue[10])
            xNeu = np.int32(CDCValue[2])*self.Xgrid[self.dIndTranRealToSim[2]] + \
                   np.int32(CDCValue[6])*self.Xgrid[self.dIndTranRealToSim[6]] + \
                   np.int32(CDCValue[10])*self.Xgrid[self.dIndTranRealToSim[10]]
            yDen = np.int32(CDCValue[7]) + np.int32(CDCValue[6]) + np.int32(CDCValue[5])
            yNeu = np.int32(CDCValue[7])*self.Ygrid[self.dIndTranRealToSim[7]] + \
                   np.int32(CDCValue[6])*self.Ygrid[self.dIndTranRealToSim[6]] + \
                   np.int32(CDCValue[5])*self.Ygrid[self.dIndTranRealToSim[5]]
        elif lMaxIndex == 7:
            xDen = np.int32(CDCValue[3]) + np.int32(CDCValue[7]) + np.int32(CDCValue[11])
            xNeu = np.int32(CDCValue[3])*self.Xgrid[self.dIndTranRealToSim[3]] + \
                   np.int32(CDCValue[7])*self.Xgrid[self.dIndTranRealToSim[7]] + \
                   np.int32(CDCValue[11])*self.Xgrid[self.dIndTranRealToSim[11]]
            yDen = np.int32(CDCValue[7]) + np.int32(CDCValue[6])
            yNeu = np.int32(CDCValue[7])*self.Ygrid[self.dIndTranRealToSim[7]] + \
                   np.int32(CDCValue[6])*self.Ygrid[self.dIndTranRealToSim[6]]
        elif lMaxIndex == 8:
            xDen = np.int32(CDCValue[4]) + np.int32(CDCValue[8])
            xNeu = np.int32(CDCValue[4])*self.Xgrid[self.dIndTranRealToSim[4]] + \
                   np.int32(CDCValue[8])*self.Xgrid[self.dIndTranRealToSim[8]]
            yDen = np.int32(CDCValue[9]) + np.int32(CDCValue[8])
            yNeu = np.int32(CDCValue[9])*self.Ygrid[self.dIndTranRealToSim[9]] + \
                   np.int32(CDCValue[8])*self.Ygrid[self.dIndTranRealToSim[8]]
        elif lMaxIndex == 9:
            xDen = np.int32(CDCValue[5]) + np.int32(CDCValue[9])
            xNeu = np.int32(CDCValue[5])*self.Xgrid[self.dIndTranRealToSim[5]] + \
                   np.int32(CDCValue[9])*self.Xgrid[self.dIndTranRealToSim[9]]
            yDen = np.int32(CDCValue[10]) + np.int32(CDCValue[9]) + np.int32(CDCValue[8])
            yNeu = np.int32(CDCValue[10])*self.Ygrid[self.dIndTranRealToSim[10]] + \
                   np.int32(CDCValue[ 9])*self.Ygrid[self.dIndTranRealToSim[ 9]] + \
                   np.int32(CDCValue[ 8])*self.Ygrid[self.dIndTranRealToSim[ 8]]
        elif lMaxIndex == 10:
            xDen = np.int32(CDCValue[6]) + np.int32(CDCValue[10])
            xNeu = np.int32(CDCValue[6])*self.Xgrid[self.dIndTranRealToSim[6]] + \
                   np.int32(CDCValue[10])*self.Xgrid[self.dIndTranRealToSim[10]]
            yDen = np.int32(CDCValue[11]) + np.int32(CDCValue[10]) + np.int32(CDCValue[9])
            yNeu = np.int32(CDCValue[11])*self.Ygrid[self.dIndTranRealToSim[11]] + \
                   np.int32(CDCValue[10])*self.Ygrid[self.dIndTranRealToSim[10]] + \
                   np.int32(CDCValue[ 9])*self.Ygrid[self.dIndTranRealToSim[ 9]]
        elif lMaxIndex == 11:
            xDen = np.int32(CDCValue[7]) + np.int32(CDCValue[11])
            xNeu = np.int32(CDCValue[7])*self.Xgrid[self.dIndTranRealToSim[7]] + \
                   np.int32(CDCValue[11])*self.Xgrid[self.dIndTranRealToSim[11]]
            yDen = np.int32(CDCValue[11]) + np.int32(CDCValue[10])
            yNeu = np.int32(CDCValue[11])*self.Ygrid[self.dIndTranRealToSim[11]] + \
                   np.int32(CDCValue[10])*self.Ygrid[self.dIndTranRealToSim[10]]        
        
        if (xDen != 0) and (yDen != 0):
            xTem = np.uint32(xNeu)//np.uint32(xDen)
            yTem = np.uint32(yNeu)//np.uint32(yDen)
            self.lTrace.append(self.remapping(xTem, yTem))
    def calWeiPosFull(self, CDCValue = None):
        '''
            Cacluating the touch position by all the CDC electrodes by centroid.
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        x = self.Xgrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        y = self.Ygrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        # Convert the CDC values in real order into the simulation order
        cdc = np.uint32(np.flip(CDCValue.reshape((3,4)).T, axis = 0).flatten()) 
        # Caculating the centroid
        Den = np.uint32(np.sum(np.uint32(cdc)))
        xNeu = np.uint32(np.sum(cdc*x))
        yNeu = np.uint32(np.sum(cdc*y))
        if (Den != 0):
            xTem = np.uint32(xNeu)//np.uint32(Den)
            yTem = np.uint32(yNeu)//np.uint32(Den)
            self.lTrace.append(self.remapping(xTem, yTem))
    def calWeiPosLow(self, CDCValue = None):
        '''
            Cacluating the touch position by the 8-adjoint CDC electrodes by centroid.
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        x = self.Xgrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        y = self.Ygrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        # Convert the CDC values in real order into the simulation order
        cdc = np.flip(CDCValue.reshape((3,4)).T, axis = 0).flatten()
        maxI = np.argmax(cdc)
        # Caculating the centroid
        Den = np.int32(np.sum(np.int32(cdc) * self.dLow[maxI])) 
        xNeu = np.sum(cdc*x*self.dLow[maxI])
        yNeu = np.sum(cdc*y*self.dLow[maxI])
        if (Den != 0):
            xTem = np.uint32(xNeu)//np.uint32(Den)
            yTem = np.uint32(yNeu)//np.uint32(Den)
            self.lTrace.append(self.remapping(xTem, yTem))
    def calWeiPosGau(self, CDCValue = None):
        '''
            Cacluating the touch position by the 8-adjoint CDC electrodes by centroid with Gaussian kernel.
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        x = self.Xgrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        y = self.Ygrid.flatten()[np.asarray(list(np.arange(0,12,1)))]
        # Convert the CDC values in real order into the simulation order
        cdc = np.flip(CDCValue.reshape((3,4)).T, axis = 0).flatten()
        maxI = np.argmax(cdc)
        # Caculating the centroid
        Den = np.int32(np.sum(np.int32(cdc) * self.dGaussian[maxI])) 
        xNeu = np.sum(cdc*x*self.dGaussian[maxI])
        yNeu = np.sum(cdc*y*self.dGaussian[maxI])
        if (Den != 0):
            xTem = np.uint32(xNeu)//np.uint32(Den)
            yTem = np.uint32(yNeu)//np.uint32(Den)
            self.lTrace.append(self.remapping(xTem, yTem))
    def calWeiLabel(self, CDCValue = None):
        '''
            Cacluating the touch position with CDC electrodes by centroid with influence zone which is gotten from
            image segmentation.
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.

            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        # Get the peak label of each pixel 
        Label = self.Segmentation(CDCValue = CDCValue)
#        print("Label : \n{0}".format(Label))
        # This stores the peak of different fingers with their corridinate position of infulence zone. 
        Header = {}
        for i in range(self.gchVertWidth*self.gchHoriWidth):
            lab = Label[self.SimuPos[self.dIndTranRealToSim[i]]]
            if len(Header) == 0:
                Header.update({lab: [self.SimuPos[self.dIndTranRealToSim[i]]]})
            elif lab not in Header.keys():
                # Updating header with the pixel
                Header.update({lab: [self.SimuPos[self.dIndTranRealToSim[i]]]})
            else:
                # Adding pixel into exist header
                Header[lab].append(self.SimuPos[self.dIndTranRealToSim[i]])

        Pos = []
        # Caculating position of fingers with the influence zone
        for maxPoint in Header.keys():
            CDCValue[CDCValue<self.MUTURE_THRESHOLD]=0
            if (CDCValue[self.dIndTranSimToReal[maxPoint]]>self.MUTURE_THRESHOLD):
                # This set stores the position of the influence zone which is gotten
                # from the image segmentation.
                set1 = set(Header[maxPoint])

                # This set stores the position of the influence zone which is gotten
                # from the neighbour of the pixel.
                set2 = set(self.Neighbour[maxPoint])

                # Finding the element in set2 which is not in set1
                setDif = set2 - set1

                # List after concating
                tmpList = Header[maxPoint]+list(setDif)

                Den = 0
                Neu = np.array([0,0])        
                
                for pos in tmpList:
                    # Simulation index
                    index = self.SimuPosToInd[pos]
#                    print("Ori : {0}".format(CDCValue[self.dIndTranSimToReal[index]]))
                    Den += CDCValue[self.dIndTranSimToReal[index]]
                    Neu += CDCValue[self.dIndTranSimToReal[index]]*self.RealPos[self.dIndTranSimToReal[index]] 
                if (Den!=0):
#                    print("Den : {0}".format(Den))
                    tmp = Neu//Den
                else:
                    tmp = Neu
                
                Pos.append(tmp)
        if (len(Pos)!=0):
            self.lTrace.append(np.asarray(Pos[0]).flatten())
    def cal1DIntpXEdge(self, CDCValue = None, maxInd = None):
        '''
            Cite: Large-scale capacitive touch panels: sensor pattern design, sampling and interpolation
            Caculating the 1D postion at the edge of boundary by the formula:
            X = 1-(0.5)*(min(A,B)/max(A,B)) ~ (0,1)
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        if maxInd in [0,1,2,3]:
            xDen = np.uint32(np.max(np.array([CDCValue[maxInd], CDCValue[maxInd+4]]))) * 2
            xNeu = np.min(np.array([CDCValue[maxInd], CDCValue[maxInd+4]]))
            xCoord = np.uint32(self.gchCinScale) * np.uint32(xNeu)//np.uint32(xDen)
        elif maxInd in [8,9,10,11]:
            xDen = np.max(np.array([CDCValue[maxInd-4], CDCValue[maxInd]])) * 2
            xNeu = np.min(np.array([CDCValue[maxInd-4], CDCValue[maxInd]]))
            xCoord = np.uint32(self.gchCinScale)-np.uint32(self.gchCinScale) * np.uint32(xNeu)//np.uint32(xDen) + np.uint32(128)
        return (xDen, xNeu, xCoord)
    def cal1DIntpYEdge(self, CDCValue = None, maxInd = None):
        '''
            Cite: Large-scale capacitive touch panels: sensor pattern design, sampling and interpolation
            Caculating the 1D postion at the edge of boundary by the formula:
            X = 1-(0.5)*(min(A,B)/max(A,B)) ~ (0,1)
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        if maxInd in [3,7,11]:
            yDen = np.max(np.array([CDCValue[maxInd], CDCValue[maxInd-1]])) * 2
            yNeu = np.min(np.array([CDCValue[maxInd], CDCValue[maxInd-1]]))
            yCoord = np.uint32(self.gchCinScale) * np.uint32(yNeu)//np.uint32(yDen)
        elif maxInd in [0,4,8]:
            yDen = np.max(np.array([CDCValue[maxInd+1], CDCValue[maxInd]])) * 2
            yNeu = np.min(np.array([CDCValue[maxInd+1], CDCValue[maxInd]]))
            yCoord = np.uint32(self.gchCinScale)-np.uint32(self.gchCinScale) * np.uint32(yNeu)//np.uint32(yDen) + np.uint32(192)
        return (yDen, yNeu, yCoord)
    def cal1DIntpCM(self, CDCValue = None):
        '''
            Cite: Large-scale capacitive touch panels: sensor pattern design, sampling and interpolation
            Caculating the 1D postion at the edge of boundary by the formula:
            frac{r_{i+1}-r_{i-1}}{r_{i-1}+r_{i}+r_{i+1}} -0.5~0.5
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        maxInd = np.argmax(CDCValue)
        xDen = 0
        xNeu = 0
        xCoord = 0
        if maxInd in self.XEdge:
            xDen, xNeu, xCoord = self.cal1DIntpXEdge(CDCValue = CDCValue, maxInd = maxInd)
        else:
            # maxIndex isn't at the edge of boundary.
            xDen = np.sum(np.array([CDCValue[maxInd-4], 
                                    CDCValue[maxInd], 
                                    CDCValue[maxInd+4]]))
            '''
                Example:
                    If maxInd = 5, then r_{i} = 5, r_{i+1} = 9, r_{i-1} = 4.
            '''
            xNeu = np.int32(CDCValue[maxInd+4])-np.int32(CDCValue[maxInd-4])
            xCoord = np.uint32(96) + np.uint32(self.gchCinScale)*xNeu//xDen
            
        yDen = 0
        yNeu = 0
        yCoord = 0
        if maxInd in self.YEdge:
            yDen, yNeu, yCoord = self.cal1DIntpYEdge(CDCValue, maxInd)
        else:
            yDen = np.sum(np.array([CDCValue[maxInd-1], 
                                    CDCValue[maxInd], 
                                    CDCValue[maxInd+1]]))
            yNeu = np.int32(CDCValue[maxInd-1])-np.int32(CDCValue[maxInd+1])
            yCoord = (3-(maxInd%4))*np.uint32(64)+np.uint(32) + np.uint32(self.gchCinScale)*yNeu//yDen
            
        self.lTrace.append(np.array([xCoord, yCoord]))
    def cal1DIntpLI(self, CDCValue = None):
        '''
            Cite: Large-scale capacitive touch panels: sensor pattern design, sampling and interpolation
            Caculating the 1D postion at the edge of boundary by the formula:
            0.5*frac{r_{i+1}-r_{i-1}}{r_{i}-\min{r_{i-1},r_{i+1}}} -0.5~0.5
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        maxInd = np.argmax(CDCValue)
        xDen = 0
        xNeu = 0
        xCoord = 0
        if maxInd in self.XEdge:
            xDen, xNeu, xCoord = self.cal1DIntpXEdge(CDCValue = CDCValue, maxInd = maxInd)
        else:
            '''
                Example:
                    If maxInd = 5, then r_{i} = 5, r_{i+1} = 9, r_{i-1} = 4.
            '''
            xDen = np.int32(CDCValue[maxInd])-np.min(np.array([CDCValue[maxInd-4],CDCValue[maxInd+4]]))
            xNeu = np.int32(CDCValue[maxInd+4])-np.int32(CDCValue[maxInd-4])
            xCoord = np.uint32(96) + np.uint32(self.gchCinScale)*xNeu//np.uint32(xDen*2)
            
        yDen = 0
        yNeu = 0
        yCoord = 0
        if maxInd in self.YEdge:
            yDen, yNeu, yCoord = self.cal1DIntpYEdge(CDCValue, maxInd)
        else:
            yDen = np.int32(CDCValue[maxInd])-np.min(np.array([CDCValue[maxInd-1],CDCValue[maxInd+1]]))
            yNeu = np.int32(CDCValue[maxInd-1])-np.int32(CDCValue[maxInd+1])
            yCoord = (3-(maxInd%4))*np.uint32(64)+np.uint(32) + np.uint32(self.gchCinScale)*yNeu//np.uint32(yDen*2)
            
        self.lTrace.append(np.array([xCoord, yCoord]))        
    def cal1DIntpPA(self, CDCValue = None):
        '''
            Cite: Large-scale capacitive touch panels: sensor pattern design, sampling and interpolation
            Caculating the 1D postion at the edge of boundary by the formula:
            0.5*frac{r_{i-1}-r_{i+1}}{r_{i-1}-2*r_{i}+r_{i+1}}} -0.5~0.5
            + CDCValue: numpy array with shape (1,12) which is in the real CDC order.
            Order of simulation(Order of real data)
            0(3)  1(7)  2(11)
            3(2)  4(6)  5(10)
            6(1)  7(5)  8( 9)
            9(0) 10(4) 11( 8)
        '''
        maxInd = np.argmax(CDCValue)
        xDen = 0
        xNeu = 0
        xCoord = 0
        if maxInd in self.XEdge:
            xDen, xNeu, xCoord = self.cal1DIntpXEdge(CDCValue = CDCValue, maxInd = maxInd)
        else:
            '''
                Example:
                    If maxInd = 5, then r_{i} = 5, r_{i+1} = 9, r_{i-1} = 4.
            '''
            xDen = np.int32(CDCValue[maxInd-4])+np.int32(CDCValue[maxInd+4])-2*np.int32(CDCValue[maxInd])
            xNeu = np.int32(CDCValue[maxInd-4])-np.int32(CDCValue[maxInd+4])
            xCoord = np.uint32(96) + np.uint32(self.gchCinScale)*xNeu//np.uint32(xDen*2)
            
        yDen = 0
        yNeu = 0
        yCoord = 0
        if maxInd in self.YEdge:
            yDen, yNeu, yCoord = self.cal1DIntpYEdge(CDCValue, maxInd)
        else:
            yDen = np.int32(CDCValue[maxInd-1])+np.int32(CDCValue[maxInd+1])-2*np.int32(CDCValue[maxInd])
            yNeu = np.int32(CDCValue[maxInd+1])-np.int32(CDCValue[maxInd-1])
            yCoord = (3-(maxInd%4))*np.uint32(64)+np.uint(32) + np.uint32(self.gchCinScale)*yNeu//np.uint32(yDen*2)
            
        self.lTrace.append(np.array([xCoord, yCoord])) 
    def cal1DIntpGAU(self, CDCValue = None):
        maxInd = np.argmax(CDCValue)
        xDen = 0
        xNeu = 0
        xCoord = 0
        if maxInd in self.XEdge:
            '''
                Example:
                    If maxInd = 5, then r_{i} = 5, r_{i+1} = 9, r_{i-1} = 4.
            '''
            xDen, xNeu, xCoord = self.cal1DIntpXEdge(CDCValue = CDCValue, maxInd = maxInd)
        else:
            xDen = np.log2(CDCValue[maxInd-4])+ \
                   np.log2(CDCValue[maxInd+4])- \
                   2*np.log2(CDCValue[maxInd])
            xNeu = np.log2(CDCValue[maxInd-4])-np.log2(CDCValue[maxInd+4])
            xCoord = np.uint32(96) + np.uint32(self.gchCinScale)*xNeu//np.uint32(xDen*2)
            
        yDen = 0
        yNeu = 0
        yCoord = 0
        if maxInd in self.YEdge:
            yDen, yNeu, yCoord = self.cal1DIntpYEdge(CDCValue, maxInd)
        else:
            yDen = np.log2(CDCValue[maxInd-1])+ \
                   np.log2(CDCValue[maxInd+1])- \
                   2*np.log2(CDCValue[maxInd])
            yNeu = np.log2(CDCValue[maxInd+1])-np.log2(CDCValue[maxInd-1])
            yCoord = (3-(maxInd%4))*np.uint32(64)+np.uint(32) + np.uint32(self.gchCinScale)*yNeu//np.uint32(yDen*2)
            
        self.lTrace.append(np.array([xCoord, yCoord]))
    def Segmentation(self, CDCValue = None):
        '''
            Using gradient based approach for image segmentation
            Argument:
                + CDCValue: Real CDC values; If one want use the simulation for instead,
                            one needs convert the order of simulation into real order.
        '''
        # Convert the real order into simulated order
        tmp = np.flip(CDCValue.reshape((self.gchHoriWidth, self.gchVertWidth)).T, axis = 0)
        # Hill image(Source image)
        # This stores the hill which corresponds to each pixel. 
        Hill = np.int32(np.ones_like(tmp))*(self.gchHoriWidth*self.gchVertWidth)

        for i in range(self.gchHoriWidth*self.gchVertWidth):
            # Product of CDC value and the mask corresponding to the pixel i  
            pro = tmp.flatten()*self.dLow[i]
            # Mark the hill image by the maximum neighbour index of the pixel i
            Hill[self.SimuPos[i]] = np.argmax(pro)
#        # for i in range(self.gchHoriWidth*self.gchVertWidth):
        for i in range(self.gchHoriWidth*self.gchVertWidth):
            if (Hill[self.SimuPos[i]]<self.gchHoriWidth*self.gchVertWidth):
                # curPeak: Gradient of current pixel points to the local hill
                curPeak = Hill[self.SimuPos[i]]
                # prePeak
                prePeak = Hill[self.SimuPos[curPeak]]
                # If the current peak is not same as the previous peak, this means 
                # the current peak is not the highest peak. It should be updated 
                # by the value of prePeak until the curPeak and prePeak converge to
                # the maximum peak.
                while prePeak != curPeak:
                    Hill[self.SimuPos[i]] = prePeak
                    curPeak = Hill[self.SimuPos[i]]
                    prePeak = Hill[self.SimuPos[curPeak]]
        
        return Hill


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import Smoother as SM

    dataReader = CDCValue.cCDCvalueHandler(gchCinScale = 64,
                                           gchHoriWidth = 3,
                                           gchVertWidth = 4)
    dataReader.ReadValues(stPath = './RealData/Number08.csv',
                          lStage = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3',
                                    'Stage 4', 'Stage 5', 'Stage 6', 'Stage 7',
                                    'Stage 8', 'Stage 9', 'Stage 10', 'Stage 11'])
    dataReader.cdcValues = dataReader.cdcValues[61:93,:]
    Tra = cCoordTransfor(gchCinScale = 64,
                         gchHoriWidth = 3,
                         gchVertWidth = 4)
    KF = SM.Kalman(P = np.eye(2)*500,
                   Q = np.eye(2)*3,
                   R = np.eye(2)*3)
    
    
    
    

#    fig = plt.figure()
#    ims = []
#    for cdc in dataReader.cdcValues:
#        cdc = np.flip(cdc.reshape((3,4)).T, axis = 0)
#        img = plt.imshow(cdc, cmap='plasma',vmin = np.min(dataReader.min_cdcValue), vmax = np.max(dataReader.max_cdcValue), extent = [0,3,4,0])
#        plt.title(r'Raw CDC')
#        ims.append([img])
#    
#    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=0)
#    ani.save("RawCDC00_Full.gif", writer='pillow')
#    plt.show()  
    
    
    
#    for i in range(dataReader.cdcValues.shape[0]):
#        Tra.calWeiLabel(CDCValue = dataReader.cdcValues[i,:])
#        tra = np.int32(np.asarray(Tra.lTrace))

    
    fig3 = plt.figure()
    ax = plt.gca()
    ims3 = []
    for i in range(dataReader.cdcValues.shape[0]):
        method = '(8 adjoint with Gau. filter)'
        Tra.calWeiLabel(CDCValue = dataReader.cdcValues[i,:])
        tra = np.int32(np.asarray(Tra.lTrace))
        TraImg = plt.plot(tra[:,0]/64, tra[:,1]/64, 'ro')
        plt.xlim([0,192/64])
        plt.ylim([256/64,0])
        method = 'Trace '+method
        plt.title('Position estimation')
        ax.set_aspect('equal')
        ims3.append(TraImg)

    ani3 = animation.ArtistAnimation(fig3, ims3, interval=200, repeat_delay=0)
    ani3.save('Trace00_Full.gif', writer='pillow')


    