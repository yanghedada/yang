# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:52:11 2018

@author: yanghe
"""

from PyEMD import EMD
import numpy  as np
import pylab as plt
#from pyhht.visualization import plot_imfs
# Define signal
#==============================================================================
# t = np.linspace(0, 1, 200)
# s = np.cos(11*2*np.pi*t*t) + 6*t*t
# 
#==============================================================================

t = np.linspace(0, 1, 500)
sin = lambda x,p: np.sin(2*np.pi*x*t+p)
S = 3*sin(18,0.2)*(t-0.2)**2
S += 5*sin(11,2.7)
S += 3*sin(14,1.6)
S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
S += t**2.1 -t

# Execute EMD on signal
IMF = EMD().emd(S,t)
N = IMF.shape[0]+1


# Plot results

plt.subplot(N,1,1)
plt.plot(t, S, 'r')
plt.title("EMD")
plt.ylabel("Signal")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(t, imf, 'g')
    plt.ylabel("IMF "+str(n+1))
    plt.locator_params(axis='y', nbins=5)
plt.subplot(N,1,n+2)
plt.plot(t, imf, 'g')
plt.ylabel("res")
plt.locator_params(axis='y', nbins=5)
#plt.tight_layout()
plt.show()



