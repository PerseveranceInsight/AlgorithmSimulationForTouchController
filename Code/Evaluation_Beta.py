import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy


import Position as PS
import Smoother as SM

if __name__ == '__main__':
    lStage = ['Count 1','Point 1 X','Point 1 Y','Pressure 1',
              'Count 2','Point 2 X','Point 2 Y','Pressure 2', 
              'Count 3','Point 3 X','Point 3 Y','Pressure 3']
    PH = PS.cPositionHandler()
    
    # Without noise
    PH.ReadPositions(stPath = './RealData/Trace/Trace97_TYV_WithoutNoise.csv',
                     stColumnName = lStage,
                     maxFingers = 3)
    PH.ReadPositions(stPath = './RealData/Trace/Trace120_TYV.csv',
                 stColumnName = lStage,
                 maxFingers = 3)
#    
    # With noise
#    PH.ReadPositions(stPath = './RealData/Trace/Trace107_TYV_WithNoise.csv',
#                     stColumnName = lStage,
#                     maxFingers = 3)
#    
    Data = PH.Position

    rawData = Data[:,2:4]
    
    fs = 100
    fc = 1
    f = fc/fs
    A = 50
    I = 0
    rawData = Data[:,2:4] + np.random.randn(Data.shape[0],1)*I
#    rawData[:,0] = Data[:,2] + np.power(np.random.randn(Data.shape[0],1).flatten(),2)*(I**2)
#    rawData[:,1] = Data[:,3] + np.power(np.cos(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180),3)*A
#    rawData[:,0] = rawData[:,0] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
#    rawData[:,1] = rawData[:,1] + np.sin(np.arange(rawData.shape[0])*2*np.pi*f + 45*np.pi/180)*A
    TY = Data[:,0:2]
    
    Start = 0
    End = rawData.shape[0]
    
    
    KF2 = SM.AdaptiveSecondKalman()
    KF3 = SM.AdaptiveSecondKalman()
    simKalman2 = []
    
    PXHhat = []
    PYHhat = []
    UXHhat = []
    UYHhat = []
    PriPX = []
    PriPY = []
    PosPX = []
    PosPY = []
    Kx = []
    Ky = []
    Zx = []
    Zy = []
    ErrX = []
    ErrY = []
    DelDX = []
    DelDY = []
    
    
    xhat = np.zeros((2,1))
    yhat = np.zeros((2,1))
    

    for i in range(End):
        
        if i>=3:
            X = rawData[i-3:i,0]
            x = rawData[i,0]
            v = rawData[i,0] - rawData[i-1,0]
            zx = np.array([x,v])
#            xhat, px, delDx = KF2.predict(X = X, Xhat = xhat)
            xhat = KF2.steadyPredict(X = X, Xhat = xhat)
#            PXHhat.append(deepcopy(xhat.flatten()))
#            PriPX.append(deepcopy(px.flatten()))
#            Zx.append(deepcopy(zx.flatten()))
#            xhat, errX, kx, px = KF2.update(Z = zx, Print = False)
            xhat= KF2.steadyUpdate(Z = zx, Print = False)
#            UXHhat.append(deepcopy(xhat.flatten()))
#            PosPX.append(deepcopy(px.flatten()))
#            Kx.append(deepcopy(kx.flatten()))
#            ErrX.append(deepcopy(errX.flatten()))
#            DelDX.append(deepcopy(delDx.flatten()))
            

            Y = rawData[i-3:i,1]
            y = rawData[i,1]
            u = rawData[i,1] - rawData[i-1,1]
            zy = np.array([y,u])
#            Zy.append(deepcopy(zy.flatten()))
#            yhat, py, delDy = KF3.predict(X = Y, Xhat = yhat, Print = False)
            yhat = KF3.steadyPredict(X = Y, Xhat = yhat, Print = False)
#            PYHhat.append(deepcopy(yhat.flatten()))
#            PriPY.append(deepcopy(py.flatten()))
#            yhat, errY, ky, py = KF3.update(Z = zy, Print = False)
            yhat = KF3.steadyUpdate(Z = zy, Print = False)
#            UYHhat.append(deepcopy(yhat.flatten()))
#            PosPY.append(deepcopy(py.flatten()))
#            Ky.append(deepcopy(ky.flatten()))
#            ErrY.append(deepcopy(errY.flatten()))
#            DelDY.append(deepcopy(delDy.flatten()))
            
            cor = np.array([xhat[0].flatten(),yhat[0].flatten()]).flatten()
            simKalman2.append(cor)
        elif i>0:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], rawData[i,0]-rawData[i-1,0]]).reshape((2,1))
            yhat = np.array([rawData[i,1], rawData[i,1]-rawData[i-1,1]]).reshape((2,1))
            PXHhat.append(deepcopy(xhat.flatten()))
            PYHhat.append(deepcopy(yhat.flatten()))
            UXHhat.append(deepcopy(xhat.flatten()))
            UYHhat.append(deepcopy(yhat.flatten()))
            Zx.append(np.asarray(xhat.flatten()))
            Zy.append(np.asarray(yhat.flatten()))
            PriPX.append((np.eye(2)*10).flatten())
            PriPY.append((np.eye(2)*10).flatten())
            PosPX.append((np.eye(2)*10).flatten())
            PosPY.append((np.eye(2)*10).flatten())
            simKalman2.append(cor)
        else:
            cor = np.array([rawData[i,0],rawData[i,1]])
            xhat = np.array([rawData[i,0], 0]).reshape((2,1))
            yhat = np.array([rawData[i,1], 0]).reshape((2,1))
            PXHhat.append(np.asarray([rawData[i,0],0]))
            PYHhat.append(np.asarray([rawData[i,1],0]))
            UXHhat.append(np.asarray([rawData[i,0],0]))
            UYHhat.append(np.asarray([rawData[i,1],0]))
            Zx.append(np.asarray([rawData[i,0],0]))
            Zy.append(np.asarray([rawData[i,1],0]))
            PriPX.append((np.eye(2)*10).flatten())
            PriPY.append((np.eye(2)*10).flatten())
            PosPX.append((np.eye(2)*10).flatten())
            PosPY.append((np.eye(2)*10).flatten())
            simKalman2.append(cor)

    simKalman2 = np.int32(np.asarray(simKalman2))
#    PXHhat = np.int32(np.asarray(PXHhat))
#    PYHhat = np.int32(np.asarray(PYHhat))
#    UXHhat = np.int32(np.asarray(UXHhat))
#    UYHhat = np.int32(np.asarray(UYHhat))
#    PosPX = np.asarray(PosPX)
#    PosPY = np.asarray(PosPY)
#    PriPX = np.asarray(PriPX)
#    PriPY = np.asarray(PriPY)
#    Kx = np.asarray(Kx)
#    Ky = np.asarray(Ky)
#    DelDX = np.asarray(DelDX)
#    DelDY = np.asarray(DelDY)
#    Zx = np.asarray(Zx)
#    Zy = np.asarray(Zy)
#    
#    filePrefix = './Npy/Adaptive2ndOrder/121219/'
#    np.save(filePrefix + 'PXHhat.npy', PXHhat)
#    np.save(filePrefix + 'PYHhat.npy', PYHhat)
#    np.save(filePrefix + 'UXHhat.npy', UXHhat)
#    np.save(filePrefix + 'UYHhat.npy', UYHhat)
#    np.save(filePrefix + 'PosPX.npy', PosPX)
#    np.save(filePrefix + 'PosPY.npy', PosPY)
#    np.save(filePrefix + 'PriPX.npy', PriPX)
#    np.save(filePrefix + 'PriPY.npy', PriPY)
#    np.save(filePrefix + 'Kx.npy', Kx)
#    np.save(filePrefix + 'Ky.npy', Ky)
#    np.save(filePrefix + 'DelDX.npy', DelDX)
#    np.save(filePrefix + 'DelDY.npy', DelDY)
#    np.save(filePrefix + 'Zx.npy', Zx)
#    np.save(filePrefix + 'Zy.npy', Zy)
    
    
    
    x = []
    xB = []
    y = []
    yB = []
    
    xD = []
    yD = []
    
    for i in range(3,End-1):
        if len(x) == 0:
            Zx = rawData[i-1:i+1,0]
            Zy = rawData[i-1:i+1,1]
            x.append(KF2.ExpontialPolyNomialForm(Zx))
            y.append(KF3.ExpontialPolyNomialForm(Zy))
            xB.append(rawData[i,0])
            yB.append(rawData[i,1])
        else:
            Zx = np.array([x[i-4], rawData[i,0]])
            Zy = np.array([y[i-4], rawData[i,1]])
            Zxx = rawData[i-1:i+1,0]
            Zyy = rawData[i-1:i+1,1]
            x.append(KF2.ExpontialPolyNomialForm(Zx))
            y.append(KF3.ExpontialPolyNomialForm(Zy))
            xB.append(deepcopy(KF2.ButterWorth(Zxx, xB[i-4])))
            yB.append(deepcopy(KF3.ButterWorth(Zyy, yB[i-4])))
            xD.append(deepcopy(KF2.AdaptiveButterWorth(Zxx, xB[i-4])))
            yD.append(deepcopy(KF3.AdaptiveButterWorth(Zyy, yB[i-4])))
       
    
    xD = np.asarray(xD)
    yD = np.asarray(yD)
        
#    figure = plt.figure()
#    
#    plt.subplot(2,3,1)
#    ax1 = plt.gca()
#    plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'bo', label = 'Raw')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax1.set_aspect('equal')
#    
#    
#    plt.subplot(2,3,2)
#    ax2 = plt.gca()
#    plt.plot(TY[Start:End+1,0], TY[Start:End+1,1], 'bo', label = 'Original')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax2.set_aspect('equal')
#    
#    plt.subplot(2,3,3)
#    ax5 = plt.gca()
#    plt.plot(simKalman2[Start:End+1,0], simKalman2[Start:End+1,1], 'bo', label = 'Adaptive Kalman 2nd')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax5.set_aspect('equal')
#    
#    plt.subplot(2,3,4)
#    ax5 = plt.gca()
#    plt.plot(x, y, 'bo', label = 'Exponential')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax5.set_aspect('equal')
#    
#    plt.subplot(2,3,5)
#    ax5 = plt.gca()
#    plt.plot(xB, yB, 'bo', label = 'Butterworth')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax5.set_aspect('equal')
#    
#    plt.subplot(2,3,6)
#    ax5 = plt.gca()
#    plt.plot(xD, yD, 'bo', label = 'Adaptive Butterworth')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    plt.legend(loc = 'lower right')
#    ax5.set_aspect('equal')
#    
#    
#    
#    
#    figure = plt.figure()
#    plt.plot(rawData[Start:End+1,0], rawData[Start:End+1,1], 'cp', label = 'Raw')
#    plt.plot(x,y, 'rd', label = 'Exponel')
#    plt.plot(xB,yB, 'mh', label = 'Butterworth')
#    plt.plot(simKalman2[4:,0], simKalman2[4:,1], 'bo', label = 'Adaptive Kalman 2nd')
#    plt.plot(xD, yD, 'yH', label = 'Adaptive Butterworth')
#    plt.plot(TY[:,0], TY[:,1], 'gs', label = 'Original')
#    ax5 = plt.gca()
#    ax5.set_aspect('equal')
#    plt.legend(loc = 'lower right')
#    plt.xlim([0,192])
#    plt.ylim([256,0])
#    

    
    
    
    
#    figure = plt.figure(figsize=(10, 10))
###    subax1 = figure.add_subplot(1,2,1)
###    subax2 = figure.add_subplot(1,2,1)
###    subax3 = figure.add_subplot(2,3,3)
###    subax4 = figure.add_subplot(2,3,4)
###    subax5 = figure.add_subplot(1,2,2)
#    subax6 = figure.add_subplot(1,1,1)
#    def animate(k):
#        subax6.cla()
##        subax6.plot(rawData[k,0], rawData[k,1], 'cp', label = 'Raw')
#        subax6.plot(rawData[0:k,0], rawData[0:k,1], 'cp', label = 'Raw')
##        subax1.set_xlim(0,192)
##        subax1.set_ylim(256,0)
##        subax1.set_aspect('equal')
#        
##        subax6.cla()
##        subax6.plot(TY[k,0], TY[k,1], 'gs', label = 'Original')
#        subax6.plot(TY[0:k,0], TY[0:k,1], 'gs', label = 'Original')
##        subax2.set_xlim(0,192)
##        subax2.set_ylim(256,0)
##        subax2.set_aspect('equal')
##        
##        subax3.cla()
##        subax6.plot(x[k],y[k], 'rd', label = 'Exponel')
#        subax6.plot(x[0:k],y[0:k], 'rd', label = 'Exponel')
##        subax3.set_xlim(0,192)
##        subax3.set_ylim(256,0)
##        subax3.set_aspect('equal')
##        
##        subax4.cla()
##        subax6.plot(xB[k],yB[k], 'mv', label = 'Butterworth')
#        subax6.plot(xB[0:k],yB[0:k], 'mv', label = 'Butterworth')
##        subax4.set_xlim(0,192)
##        subax4.set_ylim(256,0)
##        subax4.set_aspect('equal')
#        
##        subax6.cla()
##        subax6.plot(simKalman2[k,0], simKalman2[k,1], 'bo', label = 'Adaptive Kalman 2nd')
#        subax6.plot(simKalman2[0:k,0], simKalman2[0:k,1], 'bo', label = 'Adaptive Kalman 2nd')
##        subax6.set_xlim(0,192)
##        subax6.set_ylim(256,0)
##        subax6.set_aspect('equal')
#        
##        subax6.cla()
##        subax6.plot(xD[k], yD[k], 'y<', label = 'Adaptive Butterworth')
#        subax6.plot(xD[0:k], yD[0:k], 'y<', label = 'Adaptive Butterworth')
#        subax6.set_xlim(0,192)
#        subax6.set_ylim(256,0)
#        subax6.set_aspect('equal')
#        subax6.legend(loc = 'lower left')
#                
#    
#    
#    ani = animation.FuncAnimation(fig = figure,
#                                  func = animate,
#                                  frames = range(3,xD.shape[0]),
#                                  interval = 1,
#                                  blit = False,
#                                  repeat = True,
#                                  repeat_delay = 10)
##    plt.show()    
##    videoWriter = animation.FFMpegFileWriter()
#    ani.save('TraceComparison2.gif', writer='pillow')
    
    
#    figure2 = plt.figure(figsize=(8, 6))
#    subax1 = figure2.add_subplot(2,3,1)
#    subax2 = figure2.add_subplot(2,3,2)
#    subax3 = figure2.add_subplot(2,3,3)
#    subax4 = figure2.add_subplot(2,3,4)
#    subax5 = figure2.add_subplot(2,3,5)
#    subax6 = figure2.add_subplot(2,3,6)
#    def animate(k):
#        subax1.cla()
##        subax1.plot(rawData[0:k,0], rawData[0:k,1], 'co', label = 'Raw')
#        subax1.plot(rawData[k,0], rawData[k,1], 'co', label = 'Raw')
#        subax1.set_xlim(0,192)
#        subax1.set_ylim(256,0)
#        subax1.set_aspect('equal')
#        subax1.set_title('Raw')
#        
#        subax2.cla()
##        subax2.plot(TY[0:k,0], TY[0:k,1], 'go', label = 'Original')
#        subax2.plot(TY[k,0], TY[k,1], 'go', label = 'Original')
#        subax2.set_xlim(0,192)
#        subax2.set_ylim(256,0)
#        subax2.set_aspect('equal')
#        subax2.set_title('Original')
#        
#        subax3.cla()
##        subax3.plot(x[0:k],y[0:k], 'ro', label = 'Exponel')
#        subax3.plot(x[k],y[k], 'ro', label = 'Exponel')
#        subax3.set_xlim(0,192)
#        subax3.set_ylim(256,0)
#        subax3.set_aspect('equal')
#        subax3.set_title('Exponel')
#        
#        subax4.cla()
##        subax4.plot(xB[0:k],yB[0:k], 'mo', label = 'Butterworth')
#        subax4.plot(xB[k],yB[k], 'mo', label = 'Butterworth')
#        subax4.set_xlim(0,192)
#        subax4.set_ylim(256,0)
#        subax4.set_aspect('equal')
#        subax4.set_title('Butterworth')
#        
#        subax5.cla()
##        subax5.plot(simKalman2[0:k,0], simKalman2[0:k,1], 'bo', label = 'Adaptive Kalman 2nd')
#        subax5.plot(simKalman2[k,0], simKalman2[k,1], 'bo', label = 'Adaptive Kalman 2nd')
#        subax5.set_xlim(0,192)
#        subax5.set_ylim(256,0)
#        subax5.set_aspect('equal')
#        subax5.set_title('Adaptive Kalman 2nd')
#        
#        subax6.cla()
##        subax6.plot(xD[0:k], yD[0:k], 'yo', label = 'Adaptive Butterworth')
#        subax6.plot(xD[k], yD[k], 'yo', label = 'Adaptive Butterworth')
#        subax6.set_xlim(0,192)
#        subax6.set_ylim(256,0)
#        subax6.set_aspect('equal')
#        subax6.set_title('Adaptive Butterworth')
##        
##                
##    
##    
#    ani = animation.FuncAnimation(fig = figure2,
#                                  func = animate,
#                                  frames = range(4,xD.shape[0]),
#                                  interval = 1,
#                                  blit = False,
#                                  repeat = True,
#                                  repeat_delay = 10)
#    ani.save('TraceComparison.gif', writer='pillow', fps = 10)
    
#    figure3 = plt.figure()
#    plt.title('Fc = {0} Hz , Fs = {1} Hz'.format(fc, fs))
#    plt.plot(np.arange(0,rawData.shape[0]/fs,0.01/fs),np.sin(np.arange(0,rawData.shape[0]/fs,0.01/fs)*2*np.pi*fc)*A, 'r-', label = 'Original signal')
#    plt.plot(np.arange(rawData.shape[0])/fs,np.sin(np.arange(rawData.shape[0])*2*np.pi*f)*A, 'b-', label = 'Sample signal')
#    plt.plot(np.arange(rawData.shape[0])/fs,np.sin(np.arange(rawData.shape[0])*2*np.pi*f)*A, 'co', label = 'Sample point')
#    plt.plot(np.arange(rawData.shape[0])/fs, np.zeros((rawData.shape[0],1)), 'gx', label = 'Sample instant')
#    plt.legend(loc = 'upper right')
    
    
    figure4 = plt.figure()
    plt.plot(np.arange(0,rawData.shape[0]), rawData[:,0], 'co-', label = 'Raw data')
    # plt.plot(np.arange(0,TY.shape[0]), TY[:,0], 'gs-', label = 'Original')
    plt.plot(np.arange(0,len(x)), x, 'rd-', label = 'Exponential')
    plt.plot(np.arange(0,len(xB)), xB, 'mH-', label = 'Butterworth')
    plt.plot(np.arange(0,simKalman2.shape[0]), simKalman2[:,0], 'bh-', label = 'Addaptive Kalman 2nd')
    plt.plot(np.arange(0,len(xD)), xD, 'y^-', label = 'Addaptive Butterworth')
    plt.legend(loc = 'upper left')
    plt.title('X displacement')
    plt.ylim([88,105])
    
    figure5 = plt.figure()
    plt.plot(np.arange(0,rawData.shape[0]), rawData[:,1], 'co-', label = 'Raw data')
    # plt.plot(np.arange(0,TY.shape[0]), TY[:,1], 'gs-', label = 'Original')
    # plt.plot(np.arange(0,len(x)), y, 'rd-', label = 'Exponential')
    plt.plot(np.arange(0,len(xB)), yB, 'mH-', label = 'Butterworth')
    plt.plot(np.arange(0,simKalman2.shape[0]), simKalman2[:,1], 'bh-', label = 'Addaptive Kalman 2nd')
    plt.plot(np.arange(0,len(xD)), yD, 'y^-', label = 'Addaptive Butterworth')
    plt.legend(loc = 'upper left')
    plt.title('Y displacement')