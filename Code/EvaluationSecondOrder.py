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
    PH.ReadPositions(stPath = './RealData/Trace/Trace68_TYV.csv',
                     stColumnName = lStage,
                     maxFingers = 3)
    
    Data = PH.Position

    rawData = Data[:,2:4]
    chipKalmanMod = Data[:,0:2]
    chipKalman = Data[:,-2:]
    
    Start = 0
    End = rawData.shape[0]
#    End = 325
    
    
    KF2 = SM.SecondKalman()
    KF3 = SM.SecondKalman()
    simKalman2 = []
    XHhat = []
    YHhat = []
    
    xhat = np.zeros((2,1))
    yhat = np.zeros((2,1))

#    for i in range(rawData.shape[0]):
    for i in range(End):
        
        if i>=3:
            X = rawData[i-3:i,0]
            x = rawData[i,0]
            v = rawData[i,0] - rawData[i-1,0]
            zx = np.array([x,v])
            xhat = KF2.steadyPredict(X = X, Xhat = xhat)
            XHhat.append(xhat[0,0])
            
            xhat = KF2.steadyUpdate(Z = zx)


            Y = rawData[i-3:i,1]
            y = rawData[i,1]
            u = rawData[i,1] - rawData[i-1,1]
            zy = np.array([y,u])
            yhat = KF3.steadyPredict(X = Y, Xhat = yhat, Print = False)
            YHhat.append(yhat[0,0])
            
            yhat = KF3.steadyUpdate(Z = zy, Print = False)
            
            cor = np.array([xhat[0].flatten(),yhat[0].flatten()]).flatten()
            simKalman2.append(cor)
        elif i>0:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], rawData[i,0]-rawData[i-1,0]]).reshape((2,1))
            yhat = np.array([rawData[i,1], rawData[i,1]-rawData[i-1,1]]).reshape((2,1))
            XHhat.append(rawData[i,0])
            YHhat.append(rawData[i,1])
            simKalman2.append(cor)
        else:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], 0]).reshape((2,1))
            yhat = np.array([rawData[i,1], 0]).reshape((2,1))
            XHhat.append(rawData[i,0])
            YHhat.append(rawData[i,1])
            simKalman2.append(cor)

    simKalman2 = np.int32(np.asarray(simKalman2))
    XHhat = np.int32(np.asarray(XHhat))
    YHhat = np.int32(np.asarray(YHhat))
#        
#    
#    
##    rawReg = LinearRegression().fit(rawData[Start:End+1,0].reshape(-1,1), rawData[Start:End+1,1].reshape(-1,1))
##    rawPredict = rawReg.predict(rawData[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(rawData[Start:End+1,1].reshape(-1,1), rawPredict)))
##    
##    rawKalman1 = LinearRegression().fit(simKalman[Start:End+1,0].reshape(-1,1), simKalman[Start:End+1,1].reshape(-1,1))
##    rawKalmanPredict = rawKalman1.predict(simKalman[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(simKalman[Start:End+1,1].reshape(-1,1), rawKalmanPredict)))
##    
##    rawBZ = LinearRegression().fit(simBezierKL[Start:End+1,0].reshape(-1,1), simBezierKL[Start:End+1,1].reshape(-1,1))
##    rawBZPredict = rawBZ.predict(simBezierKL[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(simBezierKL[Start:End+1,1].reshape(-1,1), rawBZPredict)))
##    
##    rawBZ2 = LinearRegression().fit(simBezierKL2[Start:End+1,0].reshape(-1,1), simBezierKL2[Start:End+1,1].reshape(-1,1))
##    rawBZ2Predict = rawBZ2.predict(simBezierKL2[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(simBezierKL2[Start:End+1,1].reshape(-1,1), rawBZ2Predict)))
##    
##    chipKL= LinearRegression().fit(chipKalman[Start:End+1,0].reshape(-1,1), chipKalman[Start:End+1,1].reshape(-1,1))
##    chipKLPredict = chipKL.predict(chipKalman[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(chipKalman[Start:End+1,1].reshape(-1,1), chipKLPredict)))
##    
##    simKL2= LinearRegression().fit(simKalman2[Start:End+1,0].reshape(-1,1), simKalman2[Start:End+1,1].reshape(-1,1))
##    simKalman2Predict = simKL2.predict(simKalman2[Start:End+1,0].reshape(-1,1))
##    print("mean square error: {0}".format( mean_squared_error(simKalman2[Start:End+1,1].reshape(-1,1), simKalman2Predict)))
#    
    figure = plt.figure()
    
    plt.subplot(1,3,1)
    ax1 = plt.gca()
    plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'bo', label = 'Raw')
#    plt.plot(rawData[Start:End+1,0], rawPredict, 'y', label = 'Regression')
    plt.xlim([0,192])
    plt.ylim([256,0])
    plt.legend(loc = 'lower right')
    ax1.set_aspect('equal')
    
    
    plt.subplot(1,3,2)
    ax2 = plt.gca()
    plt.plot(chipKalmanMod[Start:End+1,0], chipKalmanMod[Start:End+1,1], 'bo', label = 'Chip Kalman')
    plt.xlim([0,192])
    plt.ylim([256,0])
    plt.legend(loc = 'lower right')
    ax2.set_aspect('equal')
    
    plt.subplot(1,3,3)
    ax5 = plt.gca()
    plt.plot(simKalman2[Start:End+1,0], simKalman2[Start:End+1,1], 'bo', label = 'Kalman 2nd')
#    plt.plot(chipKalman[Start:End+1,0], chipKLPredict, 'y', label = 'Regression')
    plt.xlim([0,192])
    plt.ylim([256,0])
#    plt.xlim([100,150])
#    plt.ylim([100,50])
    plt.legend(loc = 'lower right')
    ax5.set_aspect('equal')
    
#    plt.subplot(1,2,1)
#    ax4 = plt.gca()
#    plt.plot(simKalman2[Start:End+1,0], simKalman2[Start:End+1,1], 'bo', label = 'Kalman 2')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax4.set_aspect('equal')
#    
#    
#    plt.subplot(1,2,2)
#    ax5 = plt.gca()
#    plt.plot(chipKalman[Start:End+1,0], chipKalman[Start:End+1,1], 'ro', label = 'Chip KL')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax5.set_aspect('equal')