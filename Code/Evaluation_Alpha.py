import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import Position as PS

if __name__ == '__main__':
    lStage = ['Count 1','Point 1 X','Point 1 Y','Pressure 1']
    PH = PS.cPositionHandler()
    PH.ReadPositions(stPath = './RealData/Trace/Trace30_TY.csv',
                     stColumnName = lStage,
                     maxFingers = 3)
    
    Data = PH.Position
    
    rawData = Data[:,0:2]
    
    Start = 50
    End = 300
    rawReg = LinearRegression().fit(rawData[Start:End+1,0].reshape(-1,1), rawData[Start:End+1,1].reshape(-1,1))
    rawPredict = rawReg.predict(rawData[Start:End+1,0].reshape(-1,1))
    print("mean square error: {0}".format( mean_squared_error(rawData[Start:End+1,1].reshape(-1,1), rawPredict)))
    
    figure = plt.figure()
    ax1 = plt.gca()
    plt.plot(rawData[:,0]/64, rawData[:,1]/64, 'bo', label = 'Raw')
    plt.xlim([0,192/64])
    plt.ylim([256/64,0])
    ax1.set_aspect('equal')