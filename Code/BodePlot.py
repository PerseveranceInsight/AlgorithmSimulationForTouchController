from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    wp = 5
    ws = 30
    gpass = 15
    gstop = 30
    system = signal.iirdesign(wp, ws, gpass, gstop, fs = 100, output = 'ba', analog = False, ftype = 'butter')
    # system = signal.iirdesign(wp, ws, gpass, gstop, fs = 100, output = 'ba', analog = False, ftype = 'ellip')
    fs = 100
    a = -0.9
    alpha = 0.0142
    b = (4*alpha-(1+a))/(1+a)
    A = 1
    B = a*(1+b)-2
    C = 1-a
    Poles = np.roots([1,a+b,a*b])
    Len = np.linalg.norm(Poles[0])
#    sys1 = signal.TransferFunction([0.4, 0.0], [1,-0.6], dt = 0.01)
    # sys1 = signal.TransferFunction([0.2, 0.8], [1], dt = 0.01)
    sys1 = signal.TransferFunction([0.2, 0.0], [1, -0.8], dt = 0.01)
    sys2 = signal.TransferFunction([0.1, 0.1], [1, -0.8], dt = 0.01)
#    sys3 = signal.TransferFunction([0.195, 0.736804, -1.0161768, 0.2467442, -0.0802721], [1,-0.9179, 0, 0, 0], dt = 0.01)
    sys3 = signal.TransferFunction([0.107, 1.0634906, -1.7071208, 0.8192604, -0.2722302], [1,-0.9896, 0, 0, 0], dt = 0.01)
#    sys3 = signal.TransferFunction([ 0.7878096, -0.2593548], [1,-0.96335, 0, 0, 0], dt = 0.01)
    
    sys4 = signal.TransferFunction(system[0], system[1], dt = 0.01)
    # sys4 = signal.TransferFunction([alpha, 2*alpha, alpha],[1,a+b,a*b], dt = 0.01)
    w1, mag1, phase1 = sys1.bode()
    w2, mag2, phase2 = sys2.bode()
    w3, mag3, phase3 = sys3.bode()
    w4, mag4, phase4 = sys4.bode()
    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Magnitude')
    plt.semilogx(w1/fs, mag1, 'r', label = 'Exponential')
    plt.semilogx(w2/fs, mag2, 'g', label = 'Butterworth 2nd')
    plt.semilogx(w3/fs, mag3, 'b', label = 'Kalman')
    plt.semilogx(w4/fs, mag4, 'c', label = 'Butterworth 3rd')
    plt.xlabel('Normalized frequency (rad/s)')
    plt.ylabel('dB')
    plt.legend(loc = 'lower right')
    plt.subplot(2,1,2)
    plt.semilogx(w1/fs, phase1, 'r', label = 'Exponential')
    plt.semilogx(w2/fs, phase2, 'g', label = 'Butterworth 2nd')
    plt.semilogx(w3/fs, phase3, 'b', label = 'Kalman')
    plt.semilogx(w4/fs, phase4, 'c', label = 'Butterworth 3rd')
    plt.title('Phase')
    plt.xlabel('Normalized frequency (rad/s)')
    plt.ylabel('Degree')
    plt.legend(loc = 'lower right')
    
    
    