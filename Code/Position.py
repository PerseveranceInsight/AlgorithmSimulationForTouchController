import numpy as np
import pandas as pd
import glob

class cPositionHandler():
    def __init__(self):
        self.Path = None
        self.Files = None
        self.Position = []

    def ReadPositions(self, stPath = None, stColumnName = None, maxFingers = 1):
        self.Path = stPath
        self.Files = glob.glob(self.Path)
        for fileName in self.Files:
            self.Contents = pd.read_csv(fileName)
        self.Contents = np.asarray(self.Contents[stColumnName].values)
        if maxFingers == 2:
            self.Position = np.concatenate((self.Contents[:,1:3], self.Contents[:,-3:-1]), axis = 1)
        elif maxFingers == 3:
            self.Position = np.concatenate((self.Contents[:,1:3], self.Contents[:, 5:7], self.Contents[:,-3:-1]), axis = 1)
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    import Smoother as SM
    
    lStage = ['Count 1','Point 1 X','Point 1 Y','None 1','Count 2','Point 2 X','Point 2 Y','None 2']
    PH = cPositionHandler()
    PH.ReadPositions(stPath = './RealData./Trace/Trace12.csv', stColumnName = lStage, maxFingers = 2)
    
    l = PH.Position
    
    KF = SM.Kalman(P = np.eye(2)*10,
                   Q = np.eye(2)*0.05,
                   R = np.eye(2)*1)

    BZ = SM.QuadraticBezier()

#    nTrace = []
#    for i in range(PH.Position.shape[0]-1):
#        if i>=2:
#            X = np.asarray(nTrace)[i-2:i,:]
#            Z = PH.Position[i-2:i,-2:]
#            KF.steadyPredict(X = X.T)
#            Xhat = np.int32(KF.steadyUpdate(Z = PH.Position[i,-2:]))
# #            KF.predict(X = X.T)
##            KF.predict2(X = X.T, Z = Z.T)
##            Xhat = np.int32(KF.update(X = X.T, Z = PH.Position[i,-2:]))
#            nTrace.append(Xhat)
#        else:
#            nTrace.append(PH.Position[i,-2:])
#            
#    
#    nTrace = np.asarray(nTrace)

    # figure = plt.figure()
    # ax = plt.gca()
    # plt.plot(PH.Position[:,0]/64, PH.Position[:,1]/64, 'b')
    # plt.plot(PH.Position[:,2]/64, PH.Position[:,3]/64, 'r')
    # plt.plot(nTrace[:,0]/64, nTrace[:,1]/64, 'c')
    # plt.xlim([0,192/64])
    # plt.ylim([256/64,0])
    # ax.set_aspect('equal')

#    nTrace2 = []
#    for i in range(PH.Position.shape[0]-1):
#        if i>=2:
#            est = BZ.Interpolation(B = nTrace[i-2,:],
#                                   P = nTrace[i-1,:],
#                                   D = nTrace[i,:])
#            nTrace2.append(est)
#        else:
#            nTrace2.append(nTrace[i,:])
#
#    nTrace2 = np.asarray(nTrace2)
#
#    figure = plt.figure()
#    ax = plt.gca()
#    plt.plot(PH.Position[:,0]/64, PH.Position[:,1]/64, 'b', label = 'Raw position')
##    plt.plot(PH.Position[:,2]/64, PH.Position[:,3]/64, 'r', label = 'Kalman filter')
##    plt.plot(nTrace[:,0]/64, nTrace[:,1]/64, 'k', label = 'Kalman filter2')
#    plt.plot(nTrace2[:,0]/64, nTrace2[:,1]/64, 'c', label = 'Bezier smoother after Kalman filter')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    ax.set_aspect('equal')   
#    plt.legend(loc = 'upper right')