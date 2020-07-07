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
    # PH.ReadPositions(stPath = './RealData/Trace/Trace120_TYV.csv',
    #               stColumnName = lStage,
    #               maxFingers = 3)
#    
    # With noise
    PH.ReadPositions(stPath = './RealData/Trace/Trace107_TYV_WithNoise.csv',
                    stColumnName = lStage,
                    maxFingers = 3)
#    
    Data = PH.Position
    
    fs = 100
    fc = 220
    f = fc/fs
    A = 20
    
    I = 10
    rawData = Data[:,2:4] + np.random.multivariate_normal(mean = np.asarray([0,0]),
                                                          cov = np.asarray([[I**2,0],
                                                                            [0,I**2]]),
                                                          size = Data.shape[0])
    # rawData = np.random.multivariate_normal(mean = np.asarray([120,120]),
    #                                         cov = np.asarray([[1,0],
    #                                                           [0,1]]),
    #                                         size = Data.shape[0])
    
    rawData[:,0] = rawData[:,0] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
    rawData[:,1] = rawData[:,1] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
    
    TY = Data[:,0:2]
    
    
    Start = 1
    End = rawData.shape[0]
    
    
    KF2 = SM.AdaptiveSecondKalman()
    KF3 = SM.AdaptiveSecondKalman()
    
    
    alpha1 = 0.3
    alpha2 = 0.1
    
    xExp = [rawData[0,0]]
    yExp = [rawData[0,1]]
    alEs = []
    
    for i in range(Start, End-1):
        xHatAdE,alEx = KF2.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,0], Xhat = xExp[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 10, dis2 = 40, Print = False)
        yHatAdE,alEy = KF3.AdaptiveExpontialPolyNomial(Z = rawData[i-1:i+1,1], Xhat = yExp[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 10, dis2 = 40)
        xExp.append(xHatAdE)
        yExp.append(yHatAdE)
        alEs.append([alEx, alEy])
    
    alBs = []
    xBut = [rawData[0,0]]
    yBut = [rawData[0,1]]
    
    xBut2 = [rawData[0,0]]
    yBut2 = [rawData[0,1]]
    
    
    for i in range(Start, End-1):
        xHatAdB,alBx = KF2.AdaptiveButterworth2(Z = rawData[i-1:i+1,0], Xhat = xBut[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 10, dis2 = 40)
        yHatAdB,alBy = KF3.AdaptiveButterworth2(Z = rawData[i-1:i+1,1], Xhat = yBut[i-1], alpha1 = alpha1, alpha2 = alpha2, dis1 = 10, dis2 = 40)
        xBut.append(xHatAdB)
        yBut.append(yHatAdB)
        xBut2.append(deepcopy(KF2.AdaptiveButterWorth(Z = rawData[i-1:i+1,0], X = xBut2[i-1])))
        yBut2.append(deepcopy(KF3.AdaptiveButterWorth(Z = rawData[i-1:i+1,1], X = yBut2[i-1])))
        alBs.append([alBx, alBy])
     
    xOri = [rawData[0,0]]
    yOri = [rawData[0,1]]
    for i in range(Start, End-1):
        xOri.append(deepcopy(KF2.OriginalExpontential(Z = rawData[i,0], Xhat = xOri[i-1], alphaS = 2, alphaW = 2,Print = False)))
        yOri.append(deepcopy(KF3.OriginalExpontential(Z = rawData[i,1], Xhat = yOri[i-1], alphaS = 2, alphaW = 2)))
    
        
    alEs = np.asarray(alEs)
    alBs = np.asarray(alBs)
    Dif = np.abs(alEx-alBs)
    
    xExp = np.asarray(xExp)
    yExp = np.asarray(yExp)
    
    xBut = np.asarray(xBut)
    yBut = np.asarray(yBut)
    
    
    
    simKalman2 = []
    xhat = np.zeros((2,1))
    yhat = np.zeros((2,1))
    for i in range(End):
        if i>=3:
            # X = rawData[i-3:i,0]
            X = np.asarray(simKalman2)[i-3:i,0]
            x = rawData[i,0]
            v = rawData[i,0] - rawData[i-1,0]
            zx = np.array([x,v])
            # xhat, px, delDx = KF2.predict(X = X, Xhat = xhat)
            xhat = KF2.steadyPredict(X = X, Xhat = xhat)
            # xhat, errX, kx, px = KF2.update(Z = zx, Print = False)
            xhat= KF2.steadyUpdate(Z = zx, Print = False)

            

            # Y = rawData[i-3:i,1]
            Y = np.asarray(simKalman2)[i-3:i,1]
            y = rawData[i,1]
            u = rawData[i,1] - rawData[i-1,1]
            zy = np.array([y,u])
#            Zy.append(deepcopy(zy.flatten()))
            # yhat, py, delDy = KF3.predict(X = Y, Xhat = yhat, Print = False)
            yhat = KF3.steadyPredict(X = Y, Xhat = yhat, Print = False)
            # yhat, errY, ky, py = KF3.update(Z = zy, Print = False)
            yhat = KF3.steadyUpdate(Z = zy, Print = False)

            
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
    
    wp = 5
    ws = 30
    gpass = 7
    gstop = 30
    system = signal.iirdesign(wp, ws, gpass, gstop, fs = 100, output = 'ba', analog = False, ftype = 'butter')
    # system = signal.iirdesign(wp, ws, gpass, gstop, fs = 100, output = 'ba', analog = False, ftype = 'cheby1')
    
    # a = 0.9
    # alpha = 0.0148
    # b = (4*alpha-(1+a))/(1+a)
    # system = signal.TransferFunction([alpha, 2*alpha, alpha],[1,a+b,a*b], dt = 0.01)
    # system = (system.num, system.den)
    
    zi = signal.lfilter_zi(system[0], system[1])
    xOut = signal.lfilter(system[0], system[1], rawData[:,0], zi = zi*rawData[0,0])
    yOut = signal.lfilter(system[0], system[1], rawData[:,1], zi = zi*rawData[0,1])
    xOut = np.int32(np.asarray(xOut[0]))
    yOut = np.int32(np.asarray(yOut[0]))
    
    
    print('Expontential : {0} '.format(np.linalg.norm([np.var(xExp), np.var(yExp)],2)))
    print('Butter 2nd : {0} '.format(np.linalg.norm([np.var(xBut), np.var(yBut)],2)))
    print('Sim original : {0} '.format(np.linalg.norm([np.var(xOri), np.var(yOri)],2)))
    print('Var Butter 3rd: {0} '.format(np.linalg.norm([np.var(xOut), np.var(yOut)],2)))
    print('Kalman: {0} '.format(np.linalg.norm([np.var(simKalman2[:,0]), np.var(simKalman2[:,1])],2)))
    
        
    # figure = plt.figure()
    
    # plt.subplot(2,3,1)
    # ax1 = plt.gca()
    # plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'bo', label = 'Raw')
    # plt.plot(Data[Start:End+1,2], Data[Start:End+1,3], 'rs', label = 'Raw')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax1.set_aspect('equal')
    
    
    # plt.subplot(2,3,2)
    # ax2 = plt.gca()
    # plt.plot(xExp, yExp, 'bo', label = 'Exp')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax2.set_aspect('equal')
    
    # plt.subplot(2,3,3)
    # ax3 = plt.gca()
    # plt.plot(xBut, yBut, 'bo', label = 'Adaptive Butter')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax3.set_aspect('equal')
    
    # plt.subplot(2,3,4)
    # ax4 = plt.gca()
    # # plt.plot(simKalman2[:,0], simKalman2[:,1], 'bo', label = 'Adaptive Kalman')
    # plt.plot(xOri, yOri, 'bo', label = 'Sim Ori ')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax4.set_aspect('equal')
    
    # plt.subplot(2,3,5)
    # ax4 = plt.gca()
    # plt.plot(TY[:,0], TY[:,1], 'bo', label = 'Original')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax4.set_aspect('equal')
    
    # plt.subplot(2,3,6)
    # ax4 = plt.gca()
    # plt.plot(xOut, yOut, 'bo', label = 'Butter 3rd')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    # plt.legend(loc = 'lower right')
    # ax4.set_aspect('equal')
    
    
    
    # figure = plt.figure()
    # plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'cp-', label = 'Raw')
    # plt.plot(TY[:,0], TY[:,1], 'yh-', label = 'Original')
    # plt.plot(xOri, yOri, 'k^-', label = 'Sim original')
    # plt.plot(xExp, yExp, 'gs-', label = 'Expontential')
    # plt.plot(xBut, yBut, 'rH-', label = 'Butter 2nd')
    # plt.plot(xOut, yOut, 'bo-', label = 'Butter 3rd')
    # ax5 = plt.gca()
    # ax5.set_aspect('equal')
    # plt.legend(loc = 'lower right')
    # plt.xlim([0,192])
    # plt.ylim([256,0])
    
    # figure4 = plt.figure()
    # plt.plot(np.arange(0,rawData.shape[0]), rawData[:,0], 'co-', label = 'Raw data')
    # plt.plot(np.arange(0,TY.shape[0]), TY[:,0], 'yH-', label = 'Original')
    # plt.plot(np.arange(0,xExp.shape[0]), xExp, 'gH-', label = 'Adaptive Exp')
    # plt.plot(np.arange(0,xBut.shape[0]), xBut, 'mh-', label = 'Adaptive Butter 2nd')
    # plt.plot(np.arange(0,xOut.shape[0]), xOut, 'rd-', label = 'Butter 3rd')
    # plt.plot(np.arange(0,simKalman2.shape[0]), simKalman2[:,0], 'b^-', label = 'Adaptive Kalman')
    # plt.legend(loc = 'upper left')
    # plt.xlabel('Time')
    # plt.ylabel('Displacement')
    # plt.title('X displacement')
    # # plt.ylim([88,105])
    
    # figure5 = plt.figure()
    # plt.plot(np.arange(0,rawData.shape[0]), rawData[:,1], 'co-', label = 'Raw data')
    # plt.plot(np.arange(0,TY.shape[0]), TY[:,1], 'yH-', label = 'Original')
    # plt.plot(np.arange(0,yExp.shape[0]), yExp, 'gH-', label = 'Exponential')
    # plt.plot(np.arange(0,yBut.shape[0]), yBut, 'mh-', label = 'Adaptive Butter 2nd')
    # plt.plot(np.arange(0,yOut.shape[0]), yOut, 'rd-', label = 'Butter 3rd')
    # plt.plot(np.arange(0,simKalman2.shape[0]), simKalman2[:,1], 'b^-', label = 'Adaptive Kalman')
    # plt.xlabel('Time')
    # plt.ylabel('Displacement')
    # plt.legend(loc = 'upper left')
    # plt.title('Y displacement')
    
    
    # figure2 = plt.figure(figsize=(8, 10))
    # subax1 = figure2.add_subplot(2,3,1)
    # subax2 = figure2.add_subplot(2,3,2)
    # subax3 = figure2.add_subplot(2,3,3)
    # subax4 = figure2.add_subplot(2,3,4)
    # subax5 = figure2.add_subplot(2,3,5)
    # subax6 = figure2.add_subplot(2,3,6)
    # def animate(k):
    #     subax1.cla()
    #     subax1.plot(rawData[0:k,0], rawData[0:k,1], 'co', label = 'Raw')
    #     # subax1.plot(rawData[k,0], rawData[k,1], 'co', label = 'Raw')
    #     subax1.set_xlim(0,192)
    #     subax1.set_ylim(256,0)
    #     subax1.set_aspect('equal')
    #     subax1.set_title('Raw')
        
    #     subax2.cla()
    #     subax2.plot(TY[0:k,0],TY[0:k,1], 'yH', label = 'Original')
    #     # subax3.plot(xBut[k],yBut[k], 'ro', label = 'Butter')
    #     subax2.set_xlim(0,192)
    #     subax2.set_ylim(256,0)
    #     subax2.set_aspect('equal')
    #     subax2.set_title('Original')
        
    #     subax3.cla()
    #     subax3.plot(xExp[0:k], yExp[0:k], 'go', label = 'Exponential')
    #     # subax2.plot(xExp[k], yExp[k], 'go', label = 'Exponential')
    #     subax3.set_xlim(0,192)
    #     subax3.set_ylim(256,0)
    #     subax3.set_aspect('equal')
    #     subax3.set_title('Exponential')
        
    #     subax4.cla()
    #     subax4.plot(simKalman2[0:k,0],simKalman2[0:k,1], 'mo', label = 'Adaptive Kalman')
    #     subax4.set_xlim(0,192)
    #     subax4.set_ylim(256,0)
    #     subax4.set_aspect('equal')
    #     subax4.set_title('Adaptive Kalman')
        
    #     subax5.cla()
    #     subax5.plot(xBut[0:k],yBut[0:k], 'ro', label = 'Butter 2nd')
    #     # subax3.plot(xBut[k],yBut[k], 'ro', label = 'Butter')
    #     subax5.set_xlim(0,192)
    #     subax5.set_ylim(256,0)
    #     subax5.set_aspect('equal')
    #     subax5.set_title('Butter 2nd')
        
    #     subax6.cla()
    #     subax6.plot(xOut[0:k],yOut[0:k], 'bo', label = 'Butter 3rd')
    #     # subax3.plot(xBut[k],yBut[k], 'ro', label = 'Butter')
    #     subax6.set_xlim(0,192)
    #     subax6.set_ylim(256,0)
    #     subax6.set_aspect('equal')
    #     subax6.set_title('Butter 3rd')
        
        
                
    
    
    # ani = animation.FuncAnimation(fig = figure2,
    #                               func = animate,
    #                               frames = range(4,xExp.shape[0]),
    #                               interval = 1,
    #                               blit = False,
    #                               repeat = True,
    #                               repeat_delay = 10)
    # ani.save('TraceComparison.gif', writer='pillow', fps = 10)
    
    
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
        subax6.plot(rawData[0:k,0], rawData[0:k,1], 'cp-', label = 'Raw')
        subax6.plot(TY[0:k,0], TY[0:k,1], 'yH-', label = 'Original')
        subax6.plot(simKalman2[0:k,0], simKalman2[0:k,1], 'mD-', label = 'Kalman')
        subax6.plot(xExp[0:k], yExp[0:k], 'go-', label = 'Exponential')
        subax6.plot(xBut[0:k],yBut[0:k], 'rh-', label = 'Butter 2nd')
        subax6.plot(xOut[0:k], yOut[0:k], 'bs-', label = 'Butter 3rd')
        
        
        # subax6.plot(xExp[k], yExp[k], 'go-', label = 'Exponential')
        # subax6.plot(xBut[k],yBut[k], 'rh-', label = 'Butter')
        # plt.plot(xOut[k], yOut[k], 'bs-', label = 'Adaptive Cust')
        
        subax6.set_xlim(0,192)
        subax6.set_ylim(256,0)
        subax6.set_aspect('equal')
        subax6.legend(loc = 'lower left')
                
    
    
    ani = animation.FuncAnimation(fig = figure,
                                  func = animate,
                                  frames = range(3,xExp.shape[0]),
                                  interval = 1,
                                  blit = False,
                                  repeat = True,
                                  repeat_delay = 10)
#    plt.show()    
#    videoWriter = animation.FFMpegFileWriter()
    ani.save('TraceComparison2.gif', writer='pillow')
        
        
        
    