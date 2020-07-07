import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal
from copy import deepcopy


import Position as PS
import Smoother as SM

if __name__ == '__main__':
    lStage = ['Count 1','Point 1 X','Point 1 Y','Pressure 1',
              'Count 2','Point 2 X','Point 2 Y','Pressure 2', 
              'Count 3','Point 3 X','Point 3 Y','Pressure 3']
    PH = PS.cPositionHandler()
    
    # Without noise
    # PH.ReadPositions(stPath = './RealData/Trace/Trace93_TYV_WithoutNoise.csv',
    #                   stColumnName = lStage,
    #                   maxFingers = 3)
    PH.ReadPositions(stPath = './RealData/Trace/Trace121_TYV.csv',
                  stColumnName = lStage,
                  maxFingers = 3)
#    
    # With noise
    # PH.ReadPositions(stPath = './RealData/Trace/Trace107_TYV_WithNoise.csv',
    #                 stColumnName = lStage,
    #                 maxFingers = 3)
#    
    Data = PH.Position
    
    fs = 100
    fc = 220
    f = fc/fs
    A = 0
    
    I = 0
    rawData = Data[:,2:4] + np.random.randn(Data.shape[0],2)*I
    
    # rawData[:,0] = rawData[:,0] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
    # rawData[:,1] = rawData[:,1] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
    
    TY = Data[:,0:2]
    
    
    Start = 1
    End = rawData.shape[0]
    
    
    KF2 = SM.AdaptiveSecondKalman()
    KF3 = SM.AdaptiveSecondKalman()
    
    KF4 = SM.AdaptiveSecondKalman()
    KF5 = SM.AdaptiveSecondKalman()
    
    simKalman2 = []
    
    xExp = [rawData[0,0]]
    yExp = [rawData[0,1]]
    
    xExp2 = [rawData[0,0]]
    yExp2 = [rawData[0,1]]
    
    
    alpha1 = 0.3
    alpha2 = 0.1
    
    alpha3 = 0.9
    alpha4 = 0.9
    
    alEs = []
    alBs = []
    
    for i in range(Start, End-1):
        xHatAdE,_ = KF2.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,0], Xhat = xExp[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 5, dis2 = 10)
        yHatAdE,_ = KF3.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,1], Xhat = yExp[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 5, dis2 = 10)
        xExp.append(xHatAdE)
        yExp.append(yHatAdE)
        xHatAdE2,_ = KF4.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,0], Xhat = xExp2[i-1], alpha1 = alpha3, alpha2 = alpha4, dis1 = 5, dis2 = 10)
        yHatAdE2,_ = KF5.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,1], Xhat = yExp2[i-1], alpha1 = alpha3, alpha2 = alpha4, dis1 = 5, dis2 = 10)
        xExp2.append(xHatAdE2)
        yExp2.append(yHatAdE2)
        
        
        
        
        

    
    xExp = np.asarray(xExp)
    yExp = np.asarray(yExp)
    
    xExp2 = np.asarray(xExp2)
    yExp2 = np.asarray(yExp2)
    
    xExp3 = [rawData[0,0],xExp[1]]
    yExp3 = [rawData[0,1],yExp[1]]
    
    for i in range(2,xExp.shape[0]):
        xTemp = np.asarray(xExp3)
        yTemp = np.asarray(yExp3)
        v = rawData[i,0] - rawData[i-1,0]
        u = rawData[i,1] - rawData[i-1,1]
        delV = rawData[i,0] + rawData[i-2,0]  - rawData[i-1,0]*2
        delU = rawData[i,1] + rawData[i-2,1]  - rawData[i-1,1]*2
        
        if (np.abs(delV)>10):
            xExp3.append(xExp[i]+0.5*v + 0.2*delV)
        else:
            xExp3.append(xExp[i]+0*v)
        if (np.abs(delU)>10):
            yExp3.append(yExp[i]+0.5*u + 0.2*delU)
        else:
            yExp3.append(yExp[i]+0*u)
        
    xExp3 = np.int32(np.asarray(xExp3))
    yExp3 = np.int32(np.asarray(yExp3))
    
    # figure = plt.figure()
    
    # plt.subplot(1,3,1)
    # ax1 = plt.gca()
    # plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'bo', label = 'Raw')
    # # plt.plot(Data[Start,2], Data[Start,3], 'rs', label = 'Raw')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax1.set_aspect('equal')
    
    
    # plt.subplot(1,3,2)
    # ax2 = plt.gca()
    # plt.plot(xExp, yExp, 'bo', label = 'Origianl')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax2.set_aspect('equal')
    
    # plt.subplot(1,3,3)
    # ax3 = plt.gca()
    # plt.plot(xExp3, yExp3, 'bo', label = 'Cascade feedforward')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax3.set_aspect('equal')
    
    
    figure = plt.figure(figsize=(10, 10))
##    subax1 = figure.add_subplot(1,2,1)
##    subax2 = figure.add_subplot(1,2,1)
##    subax3 = figure.add_subplot(2,3,3)
##    subax4 = figure.add_subplot(2,3,4)
##    subax5 = figure.add_subplot(1,2,2)
    subax6 = figure.add_subplot(1,1,1)
    def animate(k):
        subax6.cla()
#        subax6.plot(rawData[k,0], rawData[k,1], 'cp', label = 'Raw')
        # subax6.plot(rawData[0:k,0], rawData[0:k,1], 'cp-', label = 'Raw')
        # subax6.plot(xExp[0:k], yExp[0:k], 'go-', label = 'Exponential')
        # subax6.plot(xBut[0:k],yBut[0:k], 'rh-', label = 'Butter')
        # plt.plot(xOut[0:k], yOut[0:k], 'bs-', label = 'Adaptive Cust')
        
        # subax6.plot(rawData[0:k,0], rawData[0:k,1], 'cp-', label = 'Raw')
        # subax6.plot(xExp[0:k], yExp[0:k], 'go-', label = 'Original')
        # subax6.plot(xExp3[0:k],yExp3[0:k], 'rh-', label = 'Cascade feedforward')
        
        subax6.plot(rawData[k-2:k,0], rawData[k-2:k,1], 'cp-', label = 'Raw')
        subax6.plot(xExp[k-2:k], yExp[k-2:k], 'go-', label = 'Original')
        subax6.plot(xExp3[k-2:k],yExp3[k-2:k], 'rh-', label = 'Cascade feedforward')
        
        subax6.set_xlim(0,192)
        subax6.set_ylim(256,0)
        subax6.set_aspect('equal')
        subax6.legend(loc = 'lower left')
                
    
    
    ani = animation.FuncAnimation(fig = figure,
                                  func = animate,
                                  frames = range(3,xExp.shape[0]),
                                  interval = 20,
                                  blit = False,
                                  repeat = True,
                                  repeat_delay = 10)
#    plt.show()    
#    videoWriter = animation.FFMpegFileWriter()
    ani.save('TraceComparison2.gif', writer='pillow')
        
    # figure4 = plt.figure()
    # plt.plot(np.arange(0,rawData.shape[0]), rawData[:,0], 'co-', label = 'Raw data')
    # # plt.plot(np.arange(0,xExp.shape[0]), xExp, 'rH-', label = 'Adaptive Exp')
    # # plt.plot(np.arange(0,xBut.shape[0]), xBut, 'yh-', label = 'Adaptive But')
    # # plt.plot(np.arange(0,xOut.shape[0]), xOut, 'gd-', label = 'Adaptive Cust')
    # plt.plot(np.arange(0,xExp.shape[0]), xExp, 'm^-', label = 'Original')
    # plt.plot(np.arange(0,xExp3.shape[0]), xExp3, 'gd-', label = 'Cascade feedforward')
    # plt.legend(loc = 'upper left')
    # plt.title('X displacement')
    # # plt.ylim([88,105])
    
    # figure5 = plt.figure()
    # plt.plot(np.arange(0,rawData.shape[0]), rawData[:,1], 'co-', label = 'Raw data')
    # plt.plot(np.arange(0,yExp.shape[0]), yExp, 'm^-', label = 'Original')
    # plt.plot(np.arange(0,yExp3.shape[0]), yExp3, 'gd-', label = 'Cascade feedforward')
    # plt.legend(loc = 'upper left')
    # plt.title('Y displacement')
