# -*- coding: utf-8 -*-
# @author: Nguyen Duy Sang
# CanTho University
# mail: ndsang@ctu.edu.vn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from scipy.integrate import quad
file_string = r'Rg1.csv'

# reads the input file 
data_set = pd.read_csv(file_string).to_numpy()

#   INTENS(Im) TEMPER(Tm) ENERGY(E) bv
initials = [
    [10968.19, 490.3409, 1.182633, 1.1]
]

n_value = len(initials)

# amplitude a, position b, width parameter c, and kinetic order bv 
def gok(x,a,b,c,bv):
    kbz = 8.617385e-5
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    return sum(gok(x, p[3*k], p[3*k+1], p[3*k+2], p[3*k+3]) for k in range(n))

# as the difference between the data and the function.
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)  
    
# defines FOM
def FOM(p, y, x, n):
     return sum(y - GaussSum(x,p,n))/sum(GaussSum(x,p,n))

# using least-squares optimization, minimize the difference between the data
# provided by experiment and the curve used to fit the function.

cnsts =  leastsq(
            residuals, 
            initials, 
            args=(
                data_set[:,1],          # y data
                data_set[:,0],          # x data
                n_value                 # n value
            )
        )[0]

# integrates the gaussian functions through gauss quadrature and saves the 
# results to a list, and each list is saved to its corresponding data file 
# in the empty 'area_sets' dictionary.

areas = dict()

for i in range(n_value):
    areas[i] = quad(
        gok,
        data_set[0,0],      # lower integration bound
        data_set[-1,0],     # upper integration bound
        args=(
            cnsts[3*i], 
            cnsts[3*i+1], 
            cnsts[3*i+2],
            cnsts[3*i+3]
        )
    )[0]

# defines the independent variable. 
x = data_set[:,0]
y =  data_set[:,1]

# Create figure window to plot data
fig = plt.figure(1, figsize=(9.5, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#sets the axis labels and parameters.
ax1.tick_params(direction = 'in', pad = 15)
#ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)

# plots the first two data sets: the raw data and the GaussSum.
ax1.plot(data_set[:,0], data_set[:,1], 'ko')
ax1.plot(x,GaussSum(x,cnsts, n_value))


ax1.fill_between(x, GaussSum(x,cnsts, n_value), alpha=0.05)

# adds a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.plot(
        x, 
        gok(
            x, 
            cnsts[3*i], 
            cnsts[3*i+1], 
            cnsts[3*i+2],
            cnsts[3*i+3]
        )
    )
# adds color a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.fill_between(
        x, 
        gok(
            x, 
            cnsts[3*i], 
            cnsts[3*i+1], 
            cnsts[3*i+2],
            cnsts[3*i+3]
        ),alpha=0.25
    )

# creates ledger for each graph
ledger = ['Data', 'Resultant']
for i in range(n_value):
    ledger.append(
        'P' + str(i+1)
    ) 

#adds the ledger to the graph.
ax1.legend(ledger)

#adds text FOM
ax1.text(0.2, 0.65, r'General order kinetic (GOK)''\nFOM = {0:0.8f}'
         .format(abs(FOM(cnsts, y, x, n_value))), transform=ax1.transAxes)

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.plot(x,residuals(cnsts, y, x, n_value))

ax2.set_xlabel('Temperature (K)',fontsize = 15)
ax2.set_ylabel('Residuals', fontsize = 15)
ax2.set_ylim(-1, 1)
ax2.set_yticks((-1, 0, 1))


fig.savefig(r'Rg1_Res_GOK.png')
plt.show()
