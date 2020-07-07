import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import Position as PS
import Smoother as SM

if __name__ == '__main__':
    lStage = ['Count 1','Point 1 X','Point 1 Y','Pressure 1',
              'Count 2','Point 2 X','Point 2 Y','Pressure 2', 
              'Count 3','Point 3 X','Point 3 Y','Pressure 3']
    PH = PS.cPositionHandler()
    PH.ReadPositions(stPath = './RealData/Trace/Trace34.csv',
                     stColumnName = lStage,
                     maxFingers = 3)
    
    Data = PH.Position
    
    KF = SM.Kalman(P = np.eye(2)*10,
                   Q = np.eye(2)*0.05,
                   R = np.eye(2)*1)

    BZ = SM.QuadraticBezier()
    
    rawData = Data[:,2:4]
    chipKalman = Data[:,-2:]
    chipBeKalman = Data[:,0:2]
    
    
    simKalman = []
    for i in range(rawData.shape[0]):
        if i>=2:
            # KF.steadyPredict(Xhat = simKalman[i-1].T,Z = rawData[i-2:i,:].T)
            delX = np.asarray(simKalman)[i-2:i,:].T
            KF.steadyPredict(Xhat = simKalman[i-1].T,delX = delX)
            Xhat = np.int32(KF.steadyUpdate(Z = rawData[i,:]))
            simKalman.append(Xhat)
        else:
            simKalman.append(rawData[i,:])
            
    
    simKalman = np.asarray(simKalman)
    
    simBezierKL = []
    for i in range(rawData.shape[0]):
        if i>=2:
            est = BZ.Interpolation1(B = simKalman[i-2,:],
                                    P = simKalman[i-1,:],
                                    D = simKalman[i,:])
            simBezierKL.append(est)
        else:
            simBezierKL.append(simKalman[i,:])

    simBezierKL = np.asarray(simBezierKL)
    
    simBezierKL2 = []
    for i in range(rawData.shape[0]):
        if i>=2:
            est = BZ.Interpolation2(B = simKalman[i-2,:],
                                    P = simKalman[i-1,:],
                                    D = simKalman[i,:])
            simBezierKL2.append(est)
        else:
            simBezierKL2.append(simKalman[i,:])

    simBezierKL2 = np.asarray(simBezierKL2)
    
    KF2 = SM.SecondKalman()
    KF3 = SM.SecondKalman()
    simKalman2 = []
    xhat = np.zeros((2,1))
    yhat = np.zeros((2,1))
    # rawData.shape[0]
    for i in range(rawData.shape[0]):
        if i>=3:
            X = rawData[i-3:i,0]
            x = rawData[i,0]
            v = rawData[i,0] - rawData[i-1,0]
            zx = np.array([x,v])
            KF2.steadyPredict(X = X, Xhat = xhat)
            xhat = KF2.steadyUpdate(Z = zx)

            Y = rawData[i-3:i,1]
            y = rawData[i,1]
            u = rawData[i,1] - rawData[i-1,1]
            zy = np.array([y,u])
            KF3.steadyPredict(X = Y, Xhat = yhat)
            yhat = KF3.steadyUpdate(Z = zy)
            cor = np.array([xhat[0].flatten(),yhat[0].flatten()]).flatten()
            simKalman2.append(cor)
        elif i>0:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], rawData[i,0]-rawData[i-1,0]]).reshape((2,1))
            yhat = np.array([rawData[i,1], rawData[i,1]-rawData[i-1,1]]).reshape((2,1))
            simKalman2.append(cor)
        else:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], 0]).reshape((2,1))
            yhat = np.array([rawData[i,1], 0]).reshape((2,1))
            simKalman2.append(cor)

    simKalman2 = np.int32(np.asarray(simKalman2))
    
    Start = 215
#    End = rawData.shape[0]
    End = 325
    
    rawReg = LinearRegression().fit(rawData[Start:End+1,0].reshape(-1,1), rawData[Start:End+1,1].reshape(-1,1))
    rawPredict = rawReg.predict(rawData[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(rawData[Start:End+1,1].reshape(-1,1), rawPredict)))
    
    rawKalman1 = LinearRegression().fit(simKalman[Start:End+1,0].reshape(-1,1), simKalman[Start:End+1,1].reshape(-1,1))
    rawKalmanPredict = rawKalman1.predict(simKalman[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(simKalman[Start:End+1,1].reshape(-1,1), rawKalmanPredict)))
    
    rawBZ = LinearRegression().fit(simBezierKL[Start:End+1,0].reshape(-1,1), simBezierKL[Start:End+1,1].reshape(-1,1))
    rawBZPredict = rawBZ.predict(simBezierKL[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(simBezierKL[Start:End+1,1].reshape(-1,1), rawBZPredict)))
    
    rawBZ2 = LinearRegression().fit(simBezierKL2[Start:End+1,0].reshape(-1,1), simBezierKL2[Start:End+1,1].reshape(-1,1))
    rawBZ2Predict = rawBZ2.predict(simBezierKL2[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(simBezierKL2[Start:End+1,1].reshape(-1,1), rawBZ2Predict)))
    
    rawKalman2 = LinearRegression().fit(simKalman2[Start:End+1,0].reshape(-1,1), simKalman2[Start:End+1,1].reshape(-1,1))
    rawKL2Predict = rawKalman2.predict(simKalman2[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(simKalman2[Start:End+1,1].reshape(-1,1), rawKL2Predict)))
    
#    figure = plt.figure()
#    plt.subplot(2,2,1)
#    ax1 = plt.gca()
#    plt.plot(rawData[Start:End+1,0]/64, rawData[Start:End+1,1]/64, 'bo', label = 'Raw')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    plt.legend(loc = 'lower right')
#    ax1.set_aspect('equal')
#    
#    plt.subplot(2,2,2)
#    ax2 = plt.gca()
#    plt.plot(chipKalman[Start:End+1,0]/64, chipKalman[Start:End+1,1]/64, 'ro', label = 'Chip KL')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    plt.legend(loc = 'lower right')
#    ax2.set_aspect('equal')
#    
#    plt.subplot(2,2,3)
#    ax3 = plt.gca()
#    plt.plot(chipBeKalman[Start:End+1,0]/64, chipBeKalman[Start:End+1,1]/64, 'go', label = 'Chip BZ')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    plt.legend(loc = 'lower right')
#    ax3.set_aspect('equal')
#    
#    plt.subplot(2,2,4)
#    ax4 = plt.gca()
#    plt.plot(simKalman2[Start:End+1,0]/64, simKalman2[Start:End+1,1]/64, 'ro', label = 'Kalman filter 2')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    plt.legend(loc = 'lower right')
#    ax4.set_aspect('equal')
    
    figure2 = plt.figure()
    ax4 = plt.gca()
    plt.plot(simKalman2[Start:End+1,0], simKalman2[Start:End+1,1], 'ro', label = 'Kalman filter 2')
    plt.xlim([0,192])
    plt.ylim([256,0])
    plt.legend(loc = 'lower right')
    ax4.set_aspect('equal')