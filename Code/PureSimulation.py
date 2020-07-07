import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import CDCValue
import CoorTransform
def ConvertSimCDCtoRealCDCFormat(sCDC):
    return np.array([sCDC[3,0], sCDC[2,0], sCDC[1,0], sCDC[0,0],
                     sCDC[3,1], sCDC[2,1], sCDC[1,1], sCDC[0,1],
                     sCDC[3,2], sCDC[2,2], sCDC[1,2], sCDC[0,2]])


if __name__ == '__main__':
    '''
        This file simulates that the finger touch across the CDC electrode. It visualizes the change of CDC 
        and its track with different method.
    '''
    Mu = np.array([3, 3])
    Sigma = np.array([[0.5,0],
                      [0,1]])
    
    PressureIntensity = 10000
        
    Simulator = CDCValue.cCDCvalueHandler(gchCinScale = 64,
                                          gchHoriWidth = 3,
                                          gchVertWidth = 4)
    Tra = CoorTransform.cCoordTransfor(gchCinScale = 64,
                                       gchHoriWidth = 3,
                                       gchVertWidth = 4)

    
    HanTra = []
    CDCs = []
    CDCtimes = []
    
#    for i in np.arange(0,3,0.1):
#        cen = np.array([i, 1])
#        HanTra.append(cen)
#        Mu = np.add(cen, np.array([np.random.randn(1)*0.0, np.random.randn(1)*0.0]))
#        Simulator.SimulationDataGenerated(npaMu = Mu[1,:],
#                                          npaSigma = Sigma,
#                                          iPreInt = PressureIntensity,
#                                          NoiseP = np.array([10,10]))
#        temp = Simulator.ConvertSimCDCtoRealCDCFormat()
#        CDCs.append(temp)


#   Two touch point
    cen1 = np.array([0, 0])
    cen2 = np.array([3, 4])
    Mu1 = np.add(cen1, np.array([np.random.randn(1)*0.0, np.random.randn(1)*0.0]))
    Mu2 = np.add(cen2, np.array([np.random.randn(1)*0.0, np.random.randn(1)*0.0]))
    Simulator.SimulationDataGenerated(npaMu = Mu1[1,:],
                                      npaSigma = Sigma,
                                      iPreInt = PressureIntensity,
                                      NoiseP = np.array([10,10]))
    temp1 = Simulator.ConvertSimCDCtoRealCDCFormat()
    Simulator.SimulationDataGenerated(npaMu = Mu2[1,:],
                                      npaSigma = Sigma,
                                      iPreInt = PressureIntensity,
                                      NoiseP = np.array([10,10]))
    temp2 = Simulator.ConvertSimCDCtoRealCDCFormat()
    CDCs.append(temp1+temp2)

    CDCs = np.asarray(CDCs)    
#    HanTra = np.asarray(HanTra)
   
#    # Display the CDC values     
#    fig = plt.figure()
#    ims = []
#    for cdc in CDCs:
#        cdc = np.flip(cdc.reshape((3,4)).T, axis = 0)
#        img = plt.imshow(cdc, cmap='plasma',vmin = np.min(CDCs), vmax = np.max(CDCs), extent = [0,3,4,0])
#        plt.title(r'Raw CDC')
#        ims.append([img])
#    
#    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=0)
#    ani.save("SimulationCDC.gif", writer='pillow')
#    plt.show()  

## Caculating raw track
#    for i in range(CDCs.shape[0]):
#            method = '(8 adjoint with Gau. filter)'
#            Tra.calWeiLabel(CDCValue = CDCs[i,:])
#            tra = np.int32(np.asarray(Tra.lTrace))


#    fig3 = plt.figure()
#    ax = plt.gca()
#    ims3 = []
#
#    for i in range(CDCs.shape[0]):    
#        TraImg = plt.plot(tra[0:i,0]/64, tra[0:i,1]/64, 'ro')
#        plt.plot(HanTra[0:i,0], HanTra[0:i,1], 'b')
#        plt.xlim([0,192/64])
#        plt.ylim([256/64,0])
#        method = 'Trace '+method
#        plt.title('Position estimation')
#        ax.set_aspect('equal')
#        ims3.append(TraImg)
#
#    ani3 = animation.ArtistAnimation(fig3, ims3, interval=200, repeat_delay=0)
#    ani3.save('SimulationRawTrack_Seg.gif', writer='pillow')
    
    
    
    
    cdc = np.flip(CDCs.reshape((3, 4)).T, axis = 0)
    pos = np.arange(12).flatten()
    pos = pos.reshape((4, 3))
    hill = Tra.Segmentation(CDCValue = CDCs)
    

    figure = plt.figure()
    X = np.array([1,2,3 ,1,2,3 ,1,2,3 ,1,2,3])
    Y = np.array([1,1,1 ,2,2,2 ,3,3,3 ,4,4,4])
    U = np.array([0,-1,-1 ,0,-1,0 ,0,1,0, 1,1,0])
    V = np.array([0,0,0 ,1,1,-1, 1,-1,-1, 0,0,0])
   
    plt.quiver(X,Y,U,V)
    plt.ylim([4,1])
    
    
    
 
