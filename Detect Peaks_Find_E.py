# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:48:47 2021

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peakdetect import peakdetect
from scipy.signal import find_peaks, peak_widths

file_string = r'Rg1.csv'
# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

x = data_set[:,0]
y = data_set[:,1]

kbz = 8.617385e-5
#bv=1.0001
# Function to calculate the TL, F1 model, T=x, maxi=a, maxt=b, engy=c
def gaussian(x,a,b,c,bv):
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))

# Generate dummy dataset
#x = np.linspace(start=350, stop=600, num=500)
#y1 = gaussian(x, 10968.19, 490.3409, 1.182633,1.001)

peaksY = peakdetect(y,x, lookahead=20) 
# Lookahead is the distance to look ahead from a peak to determine if it is the actual peak. 
# Change lookahead as necessary 
higherPeaksY = np.array(peaksY[0])
#lowerPeaks = np.array(peaks[1])
plt.plot(x,y,'bo')

#plt.plot(higherPeaksX[:,0], higherPeaksX[:,1], 'ro')
plt.plot(higherPeaksY[:,0], higherPeaksY[:,1], 'ro')

#plt.plot(lowerPeaks[:,0], lowerPeaks[:,1], 'ko')

#show results
print("Imax", higherPeaksY[:,1])
print("Tmax", higherPeaksY[:,0])
print("Imax/2", higherPeaksY[:,1]/2)

#duong thang
y_half=np.zeros_like(x)+higherPeaksY[:,1]/2
#plt.plot(x, y_half, "--", color="gray")

#tim giao diem cho nua dinh
idx = np.argwhere(np.diff(np.sign(y -y_half))).flatten()
plt.plot(x[idx], y_half[idx], 'rx')

print("T1,T2",  x[idx])
#print("T1=",x[idx][0])
#print("T2=",x[idx][1])

# Calculate E, method PS
T1=x[idx][0]
T2=x[idx][1]
Tmax=higherPeaksY[:,0]
omega=T2-T1
delta=T2-Tmax
thau=Tmax-T1
muy=delta/omega
Imax=higherPeaksY[:,0]
beta=1 # tov do gia nhiet
n=1e10 #

#FOK Lushchik and Halperin
E1H=0.976*kbz*(Tmax**2)/delta
print("E1H=",E1H)
E1L=1.51*kbz*(Tmax**2)/thau-3.16*kbz*Tmax
print("E1L=",E1L)

#SOK Lushchik and Halperin
E2H=1.81*kbz*(Tmax**2)/thau-4*kbz*Tmax
print("E2H=",E2H)
E2L=1.71*kbz*(Tmax**2)/delta
print("E2L=",E2L)

#GOK Chen
C_thau=1.51+3*(muy-0.42)
C_delta=0.976+7.3*(muy-0.42)
C_omega=2.52+10.2*(muy-0.42)
b_thau=1.58+4.2*(muy-0.42)
b_delta=0
b_omega=1
E_thau=C_thau*kbz*(Tmax**2)/thau-b_thau*2*kbz*Tmax
print("E_thau=",E_thau)
E_delta=C_delta*kbz*(Tmax**2)/delta-b_delta*2*kbz*Tmax
print("E_delta=",E_delta)
E_omega=C_omega*kbz*(Tmax**2)/omega-b_omega*2*kbz*Tmax
print("E_omega=",E_omega)

# Generate dummy dataset
#x = np.linspace(start=350, stop=600, num=500)
#y1 = gaussian(x, higherPeaksY[:,1], higherPeaksY[:,0], E_thau, b_thau )
y1 = gaussian(x, higherPeaksY[:,1], higherPeaksY[:,0], E1H, 1.1 )
plt.plot(x,y1,linewidth=3,color='green')
plt.show()