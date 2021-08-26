import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec  # unequal plots


# Function to calculate the TL, F1 model, T=x, maxi=a, maxt=b, engy=c
kbz = 8.617385e-5

# GOK
def gaussian(x,a,b,c,bv):
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))
# FOK
def gaussianFOK(x,a,b,c):
    return a*np.exp(1.0+c/kbz/x*((x-b)/b)-((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)-2.0*kbz*b/c)
# SOK
def gaussianSOK(x,a,b,c):
    bv=2
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))



# Generate dummy dataset GOK
x_dummy = np.linspace(start=350, stop=540, num=100)
y_dummy = gaussian(x_dummy, 8000, 473, 0.95, 1.61)
noise = 0.5*np.random.normal(size=y_dummy.size)
y_dummy = y_dummy + noise


# Generate dummy dataset FOK
x_dummyFOK = np.linspace(start=350, stop=540, num=100)
y_dummyFOK = gaussianFOK(x_dummyFOK, 8000, 473, 0.95)
noiseFOK = 0.5*np.random.normal(size=y_dummyFOK.size)
y_dummyFOK = y_dummyFOK + noise

# Generate dummy dataset SOK
x_dummySOK = np.linspace(start=350, stop=540, num=100)
y_dummySOK = gaussianSOK(x_dummySOK, 8000, 473, 0.95)
noiseSOK = 0.5*np.random.normal(size=y_dummySOK.size)
y_dummySOK = y_dummySOK + noise

# Create figure window to plot data GOK-MOK-OTOR
fig = plt.figure(1, figsize=(7.0, 6.0))
gs = gridspec.GridSpec(3, 1, height_ratios=[6, 6, 6])

# Top plot: data and fit ax1 for GOK
ax1 = fig.add_subplot(gs[2])
#sets the axis labels and parameters.
ax1.tick_params(direction = 'in', pad = 15)
ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)
ax1.text(0.2, 0.65, r'General order kinetic (GOK)', transform=ax1.transAxes)
# Plot the noisy exponential data
ax1.scatter(x_dummy, y_dummy, s=20, color='blue', label='Data')
# Set the axis limits
ax1.set_xlim(350, 540)
ax1.set_ylim(0, 10000)
# Fit the dummy power-law data
pars, cov = curve_fit(f=gaussian, xdata=x_dummy, ydata=y_dummy, p0=[8000, 473, 0.95, 1.61], bounds=(-np.inf, np.inf))
# Plot the noisy exponential data
ax1.scatter(x_dummy, y_dummy, s=20, color='blue', label='Data')
ax1.plot(x_dummy, gaussian(x_dummy, *pars), linewidth=2, color='green')
ax1.fill_between(x_dummy,y_dummy,alpha=0.25, color='green')
# creates ledger for each graph
ledger1 = ['Fit GOK', 'Data']
#adds the ledger to the graph.
ax1.legend(ledger1)

# Midle plot: data and fit
ax2 = fig.add_subplot(gs[1])
#sets the axis labels and parameters.
ax2.tick_params(direction = 'in', pad = 15)
ax2.set_xlabel('Temperature (K)', fontsize = 15)
ax2.set_ylabel('Intensity  (a.u.)', fontsize = 15)
ax2.text(0.2, 0.65, r'Second order kinetics (SOK)', transform=ax2.transAxes)
# Plot the noisy exponential data
ax2.scatter(x_dummySOK, y_dummySOK, s=20, color='blue', label='Data')
# Set the axis limits
ax2.set_xlim(350, 540)
ax2.set_ylim(0, 10000)
# Fit the dummy power-law data
pars, cov = curve_fit(f=gaussianSOK, xdata=x_dummySOK, ydata=y_dummySOK, p0=[8000, 473, 0.95], bounds=(-np.inf, np.inf))
# Plot the noisy exponential data
ax2.scatter(x_dummySOK, y_dummySOK, s=20, color='blue', label='Data')
ax2.plot(x_dummySOK, gaussianSOK(x_dummySOK, *pars), linewidth=2, color='red')
ax2.fill_between(x_dummySOK,y_dummySOK,alpha=0.25, color='red')
# creates ledger for each graph
ledger2 = ['Fit SOK', 'Data']
#adds the ledger to the graph.
ax2.legend(ledger2)

# Botton plot: data and fit
ax3 = fig.add_subplot(gs[0])
#sets the axis labels and parameters.
ax3.tick_params(direction = 'in', pad = 15)
ax3.set_xlabel('Temperature (K)', fontsize = 15)
ax3.set_ylabel('Intensity  (a.u.)', fontsize = 15)
ax3.text(0.15, 0.65, r'First order kinetics (FOK)', transform=ax3.transAxes)
# Set the axis limits
ax3.set_xlim(350, 540)
ax3.set_ylim(0, 10000)
# Fit the dummy power-law data
pars, cov = curve_fit(f=gaussianFOK, xdata=x_dummyFOK, ydata=y_dummyFOK, p0=[8000, 473, 0.95], bounds=(-np.inf, np.inf))
# Plot the noisy exponential data
ax3.scatter(x_dummyFOK, y_dummyFOK, s=20, color='blue', label='Data')
ax3.plot(x_dummyFOK, gaussianFOK(x_dummyFOK, *pars), linewidth=2, color='orange')
ax3.fill_between(x_dummyFOK,y_dummyFOK,alpha=0.25, color='orange')
# creates ledger for each graph
ledger3 = ['Fit FOK', 'Data']
#adds the ledger to the graph.
ax3.legend(ledger3)

plt.tight_layout()

fig.savefig(r'Sum_GOM.png')