import numpy as np

class Kalman():
    def __init__(self, P = None,
                 Q = None, R = None):
        if P is None:
            self.P = np.eye(2)*10
        
        if Q is None:
            self.Q = np.eye(2)*2

        if R is None:
            self.R = np.eye(2)*2

        self.P = P
        self.Q = Q
        self.R = R
        self.Xhat = None
    def predict(self, X):
        '''
            X: numpy array
            X: [X[k-2], X[k-1]]
        '''
        if (X.shape[1]>=2):
            F = np.array([[1,0], 
                          [0,1]])
            B = np.array([[1,   0],
                          [0,   1]])*0.7
            self.delD = X[:,1]-X[:,0]
            self.Xhat = F@X[:,1]+3*self.delD/10
            self.P = self.P+self.Q           
    def predict2(self, X, Z):
        '''
            X: numpy array
                X: [X[k-2], X[k-1]]
        '''
        if (X.shape[1]>=2):
            F = np.array([[1,0], 
                          [0,1]])
            B = np.array([[1,   0],
                          [0,   1]])*0.7
            self.delD = Z[:,1]-Z[:,0]
            self.Xhat = F@X[:,1]+3*self.delD/10
            self.P = self.P+self.Q        
    def update(self, X, Z):
        '''
            X: [X[k-2], X[k-1]], numpy array
            Z: 
        '''
        F = np.array([[1,0],
                      [0,1]])
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        B = np.array([[1,   0],
                      [0, 1]])*0.7
        err = Z - H@self.Xhat

        PHT = self.P@(H.T)

        self.S = H@PHT+self.R
        self.SI = np.linalg.inv(self.S)
        self.K = PHT@self.SI
        Xhat = self.Xhat + self.K @ err
        I_KH = np.eye(2) - self.K@H
        self.P = I_KH@self.P@(I_KH.T) + self.K@self.R@(self.K.T)
        return Xhat
    
    def steadyPredict(self, Xhat, delX):
        self.delD = delX[:,1] - delX[:,0]
        self.Xhat = Xhat + np.int16(7*self.delD/10)
        
    def steadyUpdate(self, Z):
        self.Xhat = np.int16(45*Z/100) + np.int16(55*self.Xhat/100) 
        return self.Xhat
class SecondKalman():
    def __init__(self, P = None,
                Q = None, R = None):
        if P is None:
            self.P = np.eye(2)*10
        else:
            self.P = P
        
        if Q is None:
            self.Q = np.array([[2.0,  0],
                               [0,  2.0]])
        else:
            self.Q = Q

        if R is None:
            self.R = np.array([[3.0,  0.0],
                               [0.0,  3.0]])
        else:
            self.R = R
        self.Xhat = None  
    def predict(self, X, Xhat, Print = False):
        '''
            X: numpy array
            X: [X[k-3], X[k-2], X[k-1]]
        '''
        if (X.shape[0]>=3):
            alpha = 0.3
            gamma = 0.1
            Fadding = 1.0
            F = np.array([[1,alpha], 
                          [0,    1]])
            B = np.array([[0,   0.1],
                          [0,    0]])
            d = X[2] + X[0] - 2*X[1]
#            delD = np.array([[0],[d]])
            delD = np.array([[0],[d]])
            self.Xhat = Xhat
            self.Xhat = F@self.Xhat + B@delD
            self.P = Fadding*F@(self.P)@(F.T)+gamma*self.Q
            return self.Xhat
    def update(self, Z, Print = False):
        '''
            X: 
            Z: [x,v]
        '''
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        err = np.reshape(Z,(2,1)) - H@self.Xhat

        PHT = self.P@(H.T)

        self.S = H@PHT+self.R
        self.SI = np.linalg.inv(self.S)
        self.K = PHT@self.SI
        if Print:
            print(self.K)
        self.Xhat = self.Xhat + self.K @ err
        I_KH = np.eye(2) - self.K@H
        self.P = I_KH@self.P@(I_KH.T)
        return self.Xhat
    
    def steadyPredict(self, X, Xhat, Print = False):
        '''
            X: numpy array
            X: [X[k-3], X[k-2], X[k-1]]
        '''
        if (X.shape[0]>=3):
            alpha = 0.3
            F = np.array([[1,alpha], 
                          [0,    1]])
            B = np.array([[0,   0.1],
                          [0,    0]])
            d = X[2] + X[0] - 2*X[1]
            delD = np.array([[0],[d]])
            if Print:
                print('Before predict Xhat: {0}'.format(self.Xhat))
                print('Del D : {0}'.format(delD.flatten()))
                
                
            self.Xhat = Xhat
            self.Xhat[0] = Xhat[0] + np.int16((35*Xhat[1]+10*delD[1])/100)
#            self.Xhat[0] = Xhat[0] + np.int16((2*Xhat[1]+delD[1])/10)
            self.Xhat[1] = Xhat[1]
            if Print:
                print('Predict : {0}'.format(self.Xhat.flatten()))
            return self.Xhat
    def steadyUpdate(self, Z, Print = False):
        '''
            X: 
            Z: [x,v]
        '''
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
#        if Print:
#            print('Z : {0}'.format(Z.flatten()))
        err = np.reshape(Z,(2,1)) - H@self.Xhat
        if Print:
            print('err : {0}'.format(err.flatten()))
        self.K = np.array([[0.32, 0.09],
                           [0.02, 0.3]])
    

#        self.Xhat[0] = np.int32((err[0]*36 + 9*err[1])/100) + self.Xhat[0]
#        self.Xhat[1] = np.int32((err[0]*9 + 33*err[1])/100) + self.Xhat[1]
        self.Xhat[0] = np.int32((err[0]*20 + 5*err[1])/100) + self.Xhat[0]
        self.Xhat[1] = np.int32((err[0]*1 + 18*err[1])/100) + self.Xhat[1]
#        self.Xhat[0] = np.int32((err[0]*220 + 50*err[1])/1000) + self.Xhat[0]
#        self.Xhat[1] = np.int32((err[0]*50 + 170*err[1])/1000) + self.Xhat[1]
#        if Print:
#            print('After Update : {0}'.format(self.Xhat.flatten()))
        return self.Xhat    
class AdaptiveSecondKalman():
    def __init__(self, P = None,
                Q = None, R = None):
        if P is None:
            self.P = np.eye(2)*10
        else:
            self.P = P
        
        if Q is None:
#            self.Q = np.array([[0.25,  0],
#                               [0, 2]])
            self.Q = np.array([[0.1,  0.0],
                               [0.0, 0.1]])
#            self.Q = np.array([[3.0,  0],
#                               [0,  3.0]])
        else:
            self.Q = Q

        if R is None:
            # self.R = np.array([[3,  0],
            #                    [0,  3]])
            self.R = np.array([[15, 0],
                               [0,  15]])
        else:
            self.R = R
        self.Xhat = None  
        self.Err = 0
        self.Case = []
        
    def predict(self, X, Xhat, Print = False):
        '''
            X: numpy array
            X: [X[k-3], X[k-2], X[k-1]]
        '''
        if (X.shape[0]>=3):
            # alpha = 0.3
            # gamma = 0.1
            alpha = 3
            gamma = 0.1
            Fadding = 1.0
            F = np.array([[1,alpha], 
                          [0,    1]])
            B = np.array([[0,   0.1],
                          [0,    0]])
            d = X[2] + X[0] - 2*X[1]
#            delD = np.array([[0],[d]])
            delD = np.array([[0],[d]])
            self.Xhat = Xhat
            self.Xhat = np.int32(F@self.Xhat + B@delD)
            self.P = Fadding*F@(self.P)@(F.T)+gamma*self.Q
            return self.Xhat, self.P, delD
    def update(self, Z, Print = False):
        '''
            X: 
            Z: [x,v]
        '''
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        err = np.int32(np.reshape(Z,(2,1)) - H@self.Xhat)

        PHT = self.P@(H.T)

        self.S = H@PHT+self.R
        self.SI = np.linalg.inv(self.S)
        self.K = PHT@self.SI
        if Print:
            print(self.K)
        self.Xhat = np.int32(self.Xhat + self.K @ err)
        I_KH = np.eye(2) - self.K@H
        self.P = I_KH@self.P@(I_KH.T)
        return self.Xhat, err, self.K, self.P
    
    def steadyPredict(self, X, Xhat, Print = False):
        '''
            X: numpy array
            X: [X[k-3], X[k-2], X[k-1]]
        '''
        if (X.shape[0]>=3):
            alpha = 0.3
            F = np.array([[1,alpha], 
                          [0,    1]])
            B = np.array([[0,   0.1],
                          [0,    0]])
            d = X[2] + X[0] - 2*X[1]
            delD = np.array([[0],[d]])
            if Print:
                print('Before predict Xhat: {0}'.format(self.Xhat))
                print('Del D : {0}'.format(delD.flatten()))
                 
            self.Xhat = Xhat
            self.Xhat[0] = Xhat[0] + np.int16((30*Xhat[1]+10*delD[1])/100)
#            self.Xhat[0] = Xhat[0] + np.int16((2*Xhat[1]+delD[1])/10)
            self.Xhat[1] = Xhat[1]
            if Print:
                print('Predict : {0}'.format(self.Xhat.flatten()))
            return self.Xhat
    def steadyUpdate(self, Z, Print = False):
        '''
            X: 
            Z: [x,v]
        '''
        H = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
#        if Print:
#            print('Z : {0}'.format(Z.flatten()))
        err = np.reshape(Z,(2,1)) - H@self.Xhat
#        if Print:
#            print('err : {0}'.format(err.flatten()))
        self.K = np.array([[0.32, 0.09],
                           [0.02, 0.3]])
    
#        if (np.abs(err[0])>10) and (self.Err > 10):
#            self.Case.append(1)
#            self.Xhat[0] = np.int32((err[0]*16 + 5*err[1]) /100) + self.Xhat[0]
#            self.Xhat[1] = np.int32((err[0]*5  + 11*err[1])/100) + self.Xhat[1]
#        elif ((np.abs(err[0])>10) and (self.Err < 10)) or ((np.abs(err[0])<10) and (self.Err > 10)):
#            self.Case.append(2)
#            self.Xhat[0] = np.int32((err[0]*26 + 7*err[1]) /100) + self.Xhat[0]
#            self.Xhat[1] = np.int32((err[0]*7  + 22*err[1])/100) + self.Xhat[1]
#        else:
#            self.Case.append(3)
#            self.Xhat[0] = np.int32((err[0]*36 + 9*err[1])/100) + self.Xhat[0]
#            self.Xhat[1] = np.int32((err[0]*9 + 33*err[1])/100) + self.Xhat[1]
            
        if (np.abs(err[0])>15) and (self.Err > 15):
            self.Case.append(1)
            self.Xhat[0] = np.int32((err[0]*92 + 28*err[1]) /1000) + self.Xhat[0]
            self.Xhat[1] = np.int32((err[0]*28  + 24*err[1])/1000) + self.Xhat[1]
#            self.Xhat[0] = np.int32((err[0]*231 + 59*err[1]) /1000) + self.Xhat[0]
#            self.Xhat[1] = np.int32((err[0]*59  + 199*err[1])/1000) + self.Xhat[1]
            # self.Xhat[0] = np.int32((err[0]*216 + 46*err[1]) /1000) + self.Xhat[0]
            # self.Xhat[1] = np.int32((err[0]*46  + 186*err[1])/1000) + self.Xhat[1]
        else:
            self.Case.append(2)
            # self.Xhat[0] = np.int32((err[0]*92 + 28*err[1]) /1000) + self.Xhat[0]
            # self.Xhat[1] = np.int32((err[0]*28  + 24*err[1])/1000) + self.Xhat[1]
            self.Xhat[0] = np.int32((err[0]*216 + 46*err[1]) /1000) + self.Xhat[0]
            self.Xhat[1] = np.int32((err[0]*46  + 186*err[1])/1000) + self.Xhat[1]
#        if Print:
#            print('After Update : {0}'.format(self.Xhat.flatten()))
        self.Err = np.abs(err[0])
        return self.Xhat
    
    def ExpontialPolyNomialForm(self, L):
        '''
            L[0] : X[k-1]
            L[1] : M[k]
        '''
        return np.int32(0.4*L[0] + 0.6*L[1])

    def AdaptiveExpontialPolyNomial(self, Z, Xhat, alpha1 = 0.2, alpha2 = 0.05, dis1 = 10, dis2 = 74, cinScale = 64, Print = False):
        '''
            Z: Numpy array with shape (1,2)
                Z[0] : M[k-1]
                Z[1] : M[k]
            Xhat: Real number 
                Xhat : X[k-1]


            ------dis1-----dis-----dis2-------
            ------alp1-----alp-----alp2-------
        '''
        disp = np.abs(Z[0] - Z[1])
        disp = dis2 if ( (disp)>dis2) else disp
        disp = dis1 if ( (disp)<dis1) else disp
        alpha = alpha1 + (alpha2- alpha1) * (disp - dis1)/(dis2 - dis1)
        if Print:
            print('Adp Exp Alpha : {0}'.format(1-alpha))
        return np.int32(alpha * Z[1] + (1-alpha)*Xhat), alpha
    
    def AdaptiveButterworth2(self, Z, Xhat, alpha1 = 0.3, alpha2 = 0.7, dis1 = 10, dis2 = 74, cinScale = 64, Print = False):
        '''
            Z: Numpy array with shape (1,2)
                Z[0] : M[k-1]
                Z[1] : M[k]
            Xhat: Real number 
                Xhat : X[k-1]


            ------dis1-----dis-----dis2-------
            ------alp1-----alp-----alp2-------
        '''
        disp = np.abs(Z[0] - Z[1])
        disp = dis2 if ( (disp)>dis2) else disp
        disp = dis1 if ( (disp)<dis1) else disp
        alpha = alpha1 + (alpha2- alpha1) * (disp - dis1)/(dis2 - dis1)
        if Print:
            # print((disp - dis1)/(dis2 - dis1))
            print('Alpha : {0}'.format(alpha))
        return np.int32(alpha * (Z[0]+Z[1])/2 + (1-alpha)*Xhat), alpha
    
    def OriginalExpontential(self, Z, Xhat, alphaS = 4, alphaW = 1, disS = 10, disW = 40, Print = False):
        alphaS = np.int16(63) - np.right_shift(63,np.int16(alphaS))
        alphaW = np.int16(63) - np.right_shift(63,np.int16(alphaW))
        # print('AlphaS : {0}'.format(alphaS))
        # print('AlphaW : {0}'.format(alphaW))
        D = np.abs(Z - Xhat)
        if ( (D>=disW) or (disW == disS)):
            alpha = alphaW
        elif (D<disS):
            alpha = alphaS
        else:
            temp = (alphaS - alphaW)*(D - disS)
            alpha = alphaS - np.int16(temp/(disW - disS))
        temp = (64-alpha) * Z + alpha*Xhat
        if Print:
            print('Original Alpha : {0}'.format(alpha/64))
        if Z>Xhat:
            temp += alpha
        return np.int16(temp/64)
        
        
        

    
    def ButterWorth(self, Z, X):
        return np.int32(0.8*X + 0.1*Z[0] + 0.1*Z[1] )
#        return np.int32(0.6*X + 0.2*Z[0] + 0.2*Z[1] )
#        return np.int32(0.7*X + 0.4*Z[0] - 0.1*Z[1] )
#        return np.int32(0.9*X + 0.3*Z[0] - 0.2*Z[1] )
        
    def AdaptiveButterWorth(self, Z, X):
        ratio = np.abs(Z[0]-Z[1])/64 + 1.0
#        print("Ratio : {0}".format(ratio))
        normalizeRatio = np.int32(np.abs(Z[0]-Z[1])*100/64 + 100)
#        print("Normalize ratio : {0}".format(normalizeRatio))
        alpha = ratio*0.60
        normalizeAlpha = np.int32(normalizeRatio*80/100)
#        print("Alpha : {0}".format(alpha))
        alpha = 0.87 if alpha>0.87 else alpha
        normalizeAlpha = 90 if normalizeAlpha>90 else normalizeAlpha
        # print("Normalized alpha: {0}".format(normalizeAlpha))
#        print(normalizeAlpha)
#        return np.int32(alpha*X + (1-alpha)*Z[0]/2 + (1-alpha)*Z[1]/2 )
        return np.int32(normalizeAlpha*X/100 + (100-normalizeAlpha)*Z[0]/200 + (100-normalizeAlpha)*Z[1]/200 )
#        if(np.abs(Z[0]-Z[1])>15):
#            print('Mode 1')
#            return np.int32(0.8*X + 0.1*Z[0] + 0.1*Z[1] )
#        else:
#            return np.int32(0.7*X + 0.15*Z[0] + 0.15*Z[1] )
    def Double(self, Z, X):
        a = 0.1
        b = 0.1
        print('Z : {0}'.format(Z))
        print('X : {0}'.format(X))
        print('Re : {0}'.format(-(1-b)*X[0] -(a*(1+b)-2)*X[1] - a*Z[0] + a*(1+b)*Z[1]))
        return -(1-a)*X[0] -(a*(1+b)-2)*X[1] - a*Z[0] + a*(1+b)*Z[1]

class QuadraticBezier():
    def __init__(self):
        self.B = None
        self.P = None
        self.D = None

    def Predict(self, X):
        '''
            X: numpy array
            X: [X[k-2], X[k-1]]
        '''
        self.B = X[:,1]
        vel = X[:,1] - X[:,0]
        self.P = X[:,1] + vel*0.05
    
    def Interpolation1(self, B, P, D):
        self.B = B
        self.P = P
        self.D = D
        PD = np.linalg.norm((self.P-self.D), ord = 2)
        BP = np.linalg.norm((self.P-self.B), ord = 2)
        BD = np.linalg.norm((self.D-self.B), ord = 2)
        m = np.max(np.array([BD,BP]))
        if m!= 0:
            t = 100 - np.int32(np.sqrt(PD**2 * 10000/m**2))
        else:
            t = 0
        if (t<0):
            t=0
        elif (t>100):
            t=100
        return np.int32( ((100-t)**2 * self.B + 2*(100-t)*t*self.P + (t**2)*self.D)/10000)
    
    def Interpolation2(self, B, P, D):
        self.B = B
        self.P = P
        self.D = D
        
        DP = self.D - self.P
        DB = self.B - self.D
        Dot = np.dot(DP, DB)
        if Dot!= 0:
            cosTheta = np.int32(np.abs(Dot/(np.linalg.norm(DP, ord = 2) * np.linalg.norm(DB, ord = 2)))*50)
        else:
            cosTheta = 10
        return np.int32( ((100-cosTheta)**2 * self.B + 2*(100-cosTheta)*cosTheta*self.P + (cosTheta**2)*self.D)/10000)

if __name__ == '__main__':
#    from copy import deepcopy
#    import matplotlib.pyplot as plt
#    import matplotlib.animation as animation
#    Trace = np.load('Trace.npy')
#    nTrace = deepcopy(Trace)
#    KF = Kalman(P = np.eye(2)*10,
#                Q = np.eye(2)*0.09,
#                R = np.eye(2)*1)
#    
#    for i in range(nTrace.shape[0]):
#        if i>=2:
#            measure = nTrace[i-2:i,:]
#            KF.predict(X = measure.T)
#            Xhat = KF.update(X = nTrace[i-2:i,:], Z = Trace[i,:])
#            nTrace[i,:] = Xhat
  
#    fig3 = plt.figure()
#    ax = plt.gca()
#    plt.plot(Trace[:,0]/64, Trace[:,1]/64, 'bo-', label = 'Original trace')
#    plt.plot(nTrace[:,0]/64, nTrace[:,1]/64, 'rx-', label = 'After Kalman filter')
#    plt.xlim([0,192/64])
#    plt.ylim([256/64,0])
#    ax.set_aspect('equal')
#    ax.legend()  
    QB = QuadraticBezier()
    B = np.array([160, 53])         
        
            
        
    

