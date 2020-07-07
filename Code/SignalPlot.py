from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fs = 100
    b = 0.1
    a = 0.1
    sys1 = signal.TransferFunction([0.8, 0.0], [1, -0.2], dt = 0.01)
    
    fs = 100
    fc = 5
    f = fc/fs
    A = 5
    t = np.arange(0,0.025, 1/(fs*10))
    Signal = np.sin(np.arange(t.shape[0])*2*np.pi*f)*A
    # Outputsignal = signal.lfilter(sys1.num, sys1.den, x = Signal)
    Outputsignal = np.sin(np.arange(t.shape[0])*2*np.pi*f - 50*np.pi/180)*A
    
    
    figure3 = plt.figure()
    plt.title('Fc = {0} Hz , Fs = {1} Hz'.format(fc, fs))
    plt.plot(t,Signal, 'r-', label = 'Original signal')
    plt.plot(t,Outputsignal, 'b-', label = 'Output signal')
    
    plt.legend(loc = 'upper right')

