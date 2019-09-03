import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from skleran.metrics import median_absolute_error
from skleran.metrics import label_ranking_loss
from skleran.metrics import r2_score

Dt = 0.001
t = np.arange(-10,10,Dt)
x1 = np.exp(-t**2)                          #
x2 = t*np.exp(-t**2)
x3 = (t**2-1)*np.exp(-t**2)

def wavefunction1(x):
    return np.exp(-x**2)
def wavefunction2(x):
    return x*np.exp(-x**2)
def wavefunction3(x):
    return (x**2-1)*np.exp(-x**2)


def FT(f1,g1,xmin=-10,xmax=10):
    y = np.fft.fftshift(np.fft.fft(g1))
    m = Dt*np.abs(y)
    f2 = np.arange(-len(y)/2,len(y)/2)/Dt/len(y)*(2*np.pi)


    def wavetransform(f1,xmin=-10,xmax=10,kmin=-10,kmax=10,nx=20000,nk=20000):
        k = np.linspace(kmin,kmax,nk)
        x = np.linspace(xmin,xmax,nx)
        phiReal = np.empty_like(k)
        phiImag = np.empty_like(k)
        for m in range(len(k)):
            gxreal = f1(x)*np.cos(k[m]*x)
            areaReal = np.sum(gxreal)*(xmax-xmin)/nx
            gximag = f1(x)*np.sin(k[m]*x)
            areaImag = np.sum(gximag)*(xmax-xmin)/nx
            phiReal[m] = areaReal
            phiImag[m] = areaImag
        return k, phiReal, phiImag


    k, phiReal, phiImag = wavetransform(f1,xmin,xmax)
    phi = phiReal + 1j*phiImag
    phi = np.abs(phi)
    return k, phi, m, f2

k, yalg1, m1, f2 = FT(wavefunction1,x1,xmin=-10,xmax=10)
k, yalg2, m2, f2 = FT(wavefunction2,x2,xmin=-10,xmax=10)
k, yalg3, m3, f2 = FT(wavefunction3,x3,xmin=-10,xmax=10)
ycontrol1 = np.abs(np.sqrt(np.pi) * np.exp(-k**2 / 4))
ycontrol2 = np.abs(-0.5*1j*k*np.sqrt(np.pi) * np.exp(-k**2 / 4))
ycontrol3 = np.abs(-0.25*np.sqrt(np.pi) *(k**2 +2) * np.exp(-k**2 / 4))

y_true = ycontrol1
y_pred = yalg1
