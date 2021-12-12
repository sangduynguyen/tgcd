
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from pylab import *
#from peakdetect import peakdetect

import builtins

#import array as arr

from scipy.optimize import curve_fit
from numpy import arange

# the raw string for the data file for analysis. 
# if the file is, for example, located on the desktop:

# Windows:  file_string = r'C:\Users\1mike\Desktop\Sample GC 1 pt acetone 1 pt cyclohexane.csv'
# Linux:    file_string = '/home/michael/Desktop/Sample GC 1 pt acetone 1 pt cyclohexane.csv'
# macOS:    file_string = ‘/Users/1mikegrn/Desktop/Sample GC 1 pt acetone 1 pt cyclohexane.csv’

#file_string = r'Rg1_tach.csv'
file_string = r'Rg2_tach.csv'

# reads the input file 

# if data file is in an excel file, use the following line: 
# master = pd.read_excel(file_string).to_numpy()

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

# initial guess for gaussian distributions to 
# optimize [height, position, width]. If more than 2 distributions required, 
# add a new set of [h,p,w] initial parameters to 'initials' for each new 
# distribution. New parameters should be of the same format for consistency; 
# i.e. [h,p,w],[h,p,w],[h,p,w]... etc. A 'w' guess of 1 is typically a 
# sufficient estimation.

#   INTENS(Im) TEMPER(Tm) ENERGY(E) bv
initials = [
    [391.5216, 416.4398 , 1.454349,1.00000001], 
    [565.4800, 456.4972, 1.644491,1.000000001],
   [696.7079, 483.4724, 1.704534,1.000000001],
    [1594.0088, 510.8360, 1.862064,1.00000001]
#]
#[10968.19, 490.3409, 1.182633, 1.0001]
]

        #INTENS(Im) ENERGY(E) TEMPER(Tm) bValue(b)
#1th-Peak   391.5216  1.454349   416.4398  1.210504
#2th-Peak   565.4800  1.644491   456.4972  1.379260
#3th-Peak   696.7079  1.704534   483.4724  1.085677
#4th-Peak  1594.0088  1.862064   510.8360  1.000009
# --- No changes below this line are necessary ---

# determines the number of gaussian functions 
# to compute from the initial guesses
n_value = len(initials)

# defines a typical gaussian function, of independent variable x,
# amplitude a, position b, and width parameter c.
#def gaussian(x,a,b,c):
 #   kbz = 8.617385e-5
  #  return a*np.exp(1.0+c/kbz/x*((x-b)/b)-((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)-2.0*kbz*b/c)

# defines GOK a typical gaussian function, of independent variable x,
# amplitude a, position b, width parameter c, and kinetic order bv 
def gaussian(x,a,b,c,bv):
    kbz = 8.617385e-5
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))


# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    return builtins.sum(gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3]) for k in range(n))
    #return sum(np.fromiter(gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3]) for k in range(n)))

# defines condition of minimization, called the resudual, which is defined
# as the difference between the data and the function.
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)  
    
# defines FOM
def FOM(p, y, x, n):
     return sum(y - GaussSum(x,p,n))/sum(GaussSum(x,p,n))

# E bases on PS
def E1(Tm,T1,T2):
    k = 8.617385e-5
    #return abs(1.51*((k*Tm**2)/(Tm - T1))-1.58*(2*k*Tm))
    return abs((2.52+10.2*((T2-Tm)/(T2-T1)-0.42))*((k*(Tm)**2)/(T2-T1))-(2*k*Tm))
    #return (0.976+7.3*((T2-Tm)/(T2-T1)-0.42))*((k*Tm**2)/(Tm - T1))	

'''
# E detect peak
def AE(x,a,b,c,bv):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c,bv)
    peaksY = peakdetect(y,x, lookahead=20) 
    # Lookahead is the distance to look ahead from a peak to determine if it is the actual peak. 
    # Change lookahead as necessary 
    higherPeaksY = np.array(peaksY[0])
    
    #duong thang
    y_half=np.zeros_like(x)+higherPeaksY[:,1]/2
    #plt.plot(x, y_half, "--", color="gray")
    Im=higherPeaksY[:,1]
    print("Im=",Im[0])

    #tim giao diem cho nua dinh
    idx = np.argwhere(np.diff(np.sign(y -y_half))).flatten()
    #plt.plot(x[idx], y_half[idx], 'rx')

    #print("T1,T2",  x[idx])
    # Calculate E, method PS
    T1=x[idx][0]
    T2=x[idx][1]
    Tm=higherPeaksY[:,0]
    print("T1=",T1)
    print("T2=",T2)
    print("Tm=",Tm[0])
    #omega=T2-T1
    #delta=T2-Tm
    #thau=Tm-T1
    E=E1(Tm[0],T1,T2)
    print("E=",E)
    return E
'''

#num_gen = 10
#number_of_generations = num_gen
#generations_x = []
#generations_f = []
#Im = []



# Function to calculate the TL, F1 model, T=x, maxi=a, maxt=b, engy=c
def x_function(x,a,b,c,bv):
    #a,b,c,bv=10968.19, 473, 1.1, 1.61
    kbz = 8.617385e-5
    #return a*np.exp(1.0+c/kbz/x*((x-b)/b)-((x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)-2.0*kbz*b/c)
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))

    #_function = x,y
    #return _function

# Convert decimal to a binary string
def den2bin(f):
	bStr = ''
	n = int(f)
	if n < 0: raise
	if n == 0: return '0'
	while n > 0:
		bStr = str(n % 2) + bStr
		n = n >> 1
	return bStr

#Convert decimal to a binary string of desired size of bits 
def d2b(f, b):
	n = int(f)
	base = int(b)
	ret = ""
	for y in range(base-1, -1, -1):
		ret += str((n >> y) & 1)
	return ret

#Invert Chromosome
def invchr(string, position):
	if int(string[position]) == 1:
		
		string = string[:position] + '0' + string[position+1:]
	else:
		string = string[:position] + '1' + string[position+1:]
	return string


#Roulette Wheel
def roulette(values, fitness):
	n_rand = random()*fitness
	sum_fit = 0
	for i in range(len(values)):
		sum_fit += values[i]
		if sum_fit >= n_rand:
			break
	return i	
# Func GA
def geneticTL(x,a,b,c,bv):
    # Genetic Algorithm Code to find the Maximum of F(X)
    
    #x_function(x,a,b,c,bv)
    #Range of Values
    #x_max = 32000
    x_max = 600
    #x_min = 0
    x_min = 0
    
    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 10
    number_of_generations = num_gen
    
    
    #Variables & Lists to be used during the code
    gen_1_xvalues = []
    gen_1_fvalues = []
    generations_x = []
    generations_f = []
    fitness = 0
    
    
    #Size of the string in bit
    x_size = int(len(den2bin(x_max)))
    
    #print ("Maximum size of x is", x_max,  "characters",x_max , "variables.")
    #print ("Maximum chromosome size of x is", x_size,  "bits, i.e.,", pow(2,x_size), "variables.")
    
    
    #first population - random values
    for i in range(pop_size):
    	x_tmp = int(round(randint(x_max-x_min)+x_min))
    	gen_1_xvalues.append(x_tmp)
    
    	f_tmp = x_function(x_tmp,a,b,c,bv)
    	gen_1_fvalues.append(f_tmp)
    
    	#Create total fitness
    	fitness += f_tmp
    #print ('GEN 1', gen_1_xvalues)
    
    #Getting maximum value for initial population
    max_f_gen1 = 0
    for i in range(pop_size):
    		if gen_1_fvalues[i] >= max_f_gen1:
    			max_f_gen1 = gen_1_fvalues[i]
    			max_x_gen1 = gen_1_xvalues[i]
    
    #Starting GA loop
    
    for i in range(number_of_generations):
    	#Reseting list for 2nd generation
    	gen_2_xvalues = []
    	gen_2_fvalues = []
    	selected = []
    
    	#Selecting individuals to reproduce
    	for j in range(pop_size):
    		ind_sel = roulette(gen_1_fvalues,fitness)
    		selected.append(gen_1_xvalues[ind_sel])
    
    	#Crossing the selected members
    	for j in range(0, pop_size, 2):
    		sel_ind_A = d2b(selected[j],x_size)
    		sel_ind_B = d2b(selected[j+1],x_size)
    	
    	#select point to cross over
    		cut_point = randint(1,x_size)
    	
    	#new individual AB
    		ind_AB = sel_ind_A[:cut_point] + sel_ind_B[cut_point:]
    
    	#mutation AB
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_AB, gene_position)
    			ind_AB = ind_mut
    	
    	#new individual BA
    		ind_BA = sel_ind_B[:cut_point] + sel_ind_A[cut_point:]		
    
    
    	#mutation BA
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_BA, gene_position)
    			ind_BA = ind_mut
    
    	#Creating Generation 2
    		new_AB = int(ind_AB,2)
    		gen_2_xvalues.append(new_AB)
    
    		new_f_AB = x_function(new_AB,a,b,c,bv)
    		gen_2_fvalues.append(new_f_AB)
    
    		new_BA = int(ind_BA,2)
    		gen_2_xvalues.append(new_BA)
    
    		new_f_BA = x_function(new_BA,a,b,c,bv)
    		gen_2_fvalues.append(new_f_BA)
    	#print ('GEN',i+2, gen_2_xvalues)
    
    
    	#Getting maximum value
    	max_f_gen2 = 0
    	for j in range(pop_size):
    		if gen_2_fvalues[j] >= max_f_gen2:
    			max_f_gen2 = gen_2_fvalues[j]
    			max_x_gen2 = gen_2_xvalues[j]
    
    	#Elitism one individual
    	if max_f_gen1 > max_f_gen2:
    		max_f_gen2 = max_f_gen1
    		max_x_gen2 = max_x_gen1
    		gen_2_fvalues[0] = max_f_gen1
    		gen_2_xvalues[0] = max_x_gen1
    	
    	#Transform gen2 into gen1
    	gen_1_xvalues = gen_2_xvalues
    	gen_1_fvalues = gen_2_fvalues
    	max_x_gen1 = max_x_gen2
    	max_f_gen1 = max_f_gen2
    	generations_x.append(max_x_gen2)
    	generations_f.append(max_f_gen2)
    
    	#Creating new fitness
    	fitness = 0
    	for j in range(pop_size):
    		f_tmp = x_function(gen_1_xvalues[j],a,b,c,bv)
    		fitness += f_tmp
    #print ("Max xy peak:",generations_x[num_gen-1],generations_f[num_gen-1])
    Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    return Tm
# Func GA
def geneticTLy(x,a,b,c,bv):
    # Genetic Algorithm Code to find the Maximum of F(X)
    
    #x_function(x,a,b,c,bv)
    #Range of Values
    #x_max = 32000
    x_max = 600
    #x_min = 0
    x_min = 0
    
    #GA Parameters
    # Due my laziness to do the code, the population size must be a even number and the values for x are always integers.
    # Feel free to correct it :) 
    pop_size = 200
    mutation_probability = 0.20
    num_gen = 10
    number_of_generations = num_gen
    
    
    #Variables & Lists to be used during the code
    gen_1_xvalues = []
    gen_1_fvalues = []
    generations_x = []
    generations_f = []
    fitness = 0
    
    
    #Size of the string in bit
    x_size = int(len(den2bin(x_max)))
    
    #print ("Maximum size of x is", x_max,  "characters",x_max , "variables.")
    #print ("Maximum chromosome size of x is", x_size,  "bits, i.e.,", pow(2,x_size), "variables.")
    
    
    #first population - random values
    for i in range(pop_size):
    	x_tmp = int(round(randint(x_max-x_min)+x_min))
    	gen_1_xvalues.append(x_tmp)
    
    	f_tmp = x_function(x_tmp,a,b,c,bv)
    	gen_1_fvalues.append(f_tmp)
    
    	#Create total fitness
    	fitness += f_tmp
    #print ('GEN 1', gen_1_xvalues)
    
    #Getting maximum value for initial population
    max_f_gen1 = 0
    for i in range(pop_size):
    		if gen_1_fvalues[i] >= max_f_gen1:
    			max_f_gen1 = gen_1_fvalues[i]
    			max_x_gen1 = gen_1_xvalues[i]
    
    #Starting GA loop
    
    for i in range(number_of_generations):
    	#Reseting list for 2nd generation
    	gen_2_xvalues = []
    	gen_2_fvalues = []
    	selected = []
    
    	#Selecting individuals to reproduce
    	for j in range(pop_size):
    		ind_sel = roulette(gen_1_fvalues,fitness)
    		selected.append(gen_1_xvalues[ind_sel])
    
    	#Crossing the selected members
    	for j in range(0, pop_size, 2):
    		sel_ind_A = d2b(selected[j],x_size)
    		sel_ind_B = d2b(selected[j+1],x_size)
    	
    	#select point to cross over
    		cut_point = randint(1,x_size)
    	
    	#new individual AB
    		ind_AB = sel_ind_A[:cut_point] + sel_ind_B[cut_point:]
    
    	#mutation AB
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_AB, gene_position)
    			ind_AB = ind_mut
    	
    	#new individual BA
    		ind_BA = sel_ind_B[:cut_point] + sel_ind_A[cut_point:]		
    
    
    	#mutation BA
    		ran_mut = random()
    		if ran_mut < mutation_probability:
    			gene_position = randint(0,x_size)
    			ind_mut = invchr(ind_BA, gene_position)
    			ind_BA = ind_mut
    
    	#Creating Generation 2
    		new_AB = int(ind_AB,2)
    		gen_2_xvalues.append(new_AB)
    
    		new_f_AB = x_function(new_AB,a,b,c,bv)
    		gen_2_fvalues.append(new_f_AB)
    
    		new_BA = int(ind_BA,2)
    		gen_2_xvalues.append(new_BA)
    
    		new_f_BA = x_function(new_BA,a,b,c,bv)
    		gen_2_fvalues.append(new_f_BA)
    	#print ('GEN',i+2, gen_2_xvalues)
    
    
    	#Getting maximum value
    	max_f_gen2 = 0
    	for j in range(pop_size):
    		if gen_2_fvalues[j] >= max_f_gen2:
    			max_f_gen2 = gen_2_fvalues[j]
    			max_x_gen2 = gen_2_xvalues[j]
    
    	#Elitism one individual
    	if max_f_gen1 > max_f_gen2:
    		max_f_gen2 = max_f_gen1
    		max_x_gen2 = max_x_gen1
    		gen_2_fvalues[0] = max_f_gen1
    		gen_2_xvalues[0] = max_x_gen1
    	
    	#Transform gen2 into gen1
    	gen_1_xvalues = gen_2_xvalues
    	gen_1_fvalues = gen_2_fvalues
    	max_x_gen1 = max_x_gen2
    	max_f_gen1 = max_f_gen2
    	generations_x.append(max_x_gen2)
    	generations_f.append(max_f_gen2)
    
    	#Creating new fitness
    	fitness = 0
    	for j in range(pop_size):
    		f_tmp = x_function(gen_1_xvalues[j],a,b,c,bv)
    		fitness += f_tmp
    #print ("Max xy peak:",generations_x[num_gen-1],generations_f[num_gen-1])
    #Tm = generations_x[num_gen-1]
    Im = generations_f[num_gen-1]
    return Im
# using least-squares optimization, minimize the difference between the data
# provided by experiment and the curve used to fit the function.
#def Tm_gen(x,a,b,c,bv,):
    #geneticTL(x,a,b,c,bv)
    #Tm = generations_x[num_gen-1]
    #print("T_gen=",Tm)
    #Im = generations_f[num_gen-1]

    #x=generations_x[num_gen-1]
    #Tm=generations_x[num_gen-1]
    #return Tm
#def Im_gen(x,a,b,c,bv):
    #geneticTL(x,a,b,c,bv)
    #Imi=Im
    #Tm=generations_x[num_gen-1]
    #Im=generations_f[num_gen-1]
    #print("T_gen=",Im)
    #return Im

# E bases genetic
def AE_gen(x,a,b,c,bv):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c,bv)
    Tm = geneticTL(x,a,b,c,bv)
    Im = geneticTLy(x,a,b,c,bv)
    #Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    #Tm = Tm_gen(x,a,b,c,bv)
    #Im =  Im_gen(x,a,b,c,bv)
    #Tmi=Tm
    #Imi=Im
    #Imi = Im_gen(x,a,b,c,bv)
    #peaksY = peakdetect(y,x, lookahead=20) 
    # Lookahead is the distance to look ahead from a peak to determine if it is the actual peak. 
    # Change lookahead as necessary 
    #higherPeaksY = np.array(peaksY[0])
    #Im = Im_gen(Im)
    
    #duong thang
    #y_half=np.zeros_like(x)+higherPeaksY[:,1]/2
    #y_half=np.zeros_like(x)+generations_f[num_gen-1]/2
    y_half=np.zeros_like(x)+Im/2
    
    #plt.plot(x, y_half, "--", color="gray")
    #Im=higherPeaksY[:,1]
    #Im=generations_f[num_gen-1]
    print("Img=",Im)

    #tim giao diem cho nua dinh
    idx = np.argwhere(np.diff(np.sign(y -y_half))).flatten()
    #plt.plot(x[idx], y_half[idx], 'rx')

    #print("T1,T2",  x[idx])
    # Calculate E, method PS
    T1=x[idx][0]
    T2=x[idx][1]
    #Tm=higherPeaksY[:,0]
    #Tm=generations_x[num_gen-1]
    print("T1g=",T1)
    print("T2g=",T2)
    print("Tmg=",Tm)
    #omega=T2-T1
    #delta=T2-Tm
    #thau=Tm-T1
    #E=E1(Tm[0],T1,T2)
    E=E1(Tm,T1,T2)
    print("Eg=",E)
    return E

# define the true objective function IR
def IR(x, E, b):
    return E * x + b
# E: IR
def AE_IR(x,a,b,c,bv):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c,bv)
    #Tm = geneticTL(x,a,b,c,bv)
    Im = geneticTLy(x,a,b,c,bv)
    #Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    
    #add IR
    #j=15
    j=10
    Tci=[]
    Ici=[]
    
    for i in range(1,j):
        #Im[i]=geneticTLy(x,a,b,c,bv)
        yi=np.zeros_like(x)+Im*i/100
        idx = np.argwhere(np.diff(np.sign(y - yi))).flatten()
        Tc = x[idx][0]
        Ic = y[idx][0]
        #print("Tc,Ic=",Tc,Ic)
        Tci=np.append(Tci, Tc)
        Ici=np.append(Ici, Ic)
    
    
    kbz = 8.617385e-5
    # ND
    x_ND=1/(kbz*Tci)
    
    # ln(TL)
    y_ln=np.log(Ici)
    
    # curve fit
    popt, _ = curve_fit(IR, x_ND, y_ln)
    # summarize the parameter values
    E_IR, b_IR = popt
    print("E & b:",-E_IR,b_IR)
    print('y = %.5f * x + %.5f' % (E_IR,b_IR))
    
    #E=E2(Tc)
    print("E_IR=",-E_IR)
    
    #define function to calculate adjusted r-squared
    #def R2(x1_ND, y1_ln, degree):
    #results = {}
    coeffs = np.polyfit(x_ND, y_ln, 1)
    p = np.poly1d(coeffs)
    yhat = p(x_ND)
    ybar = np.sum(y_ln)/len(y_ln)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y_ln - ybar)**2)
    R2 = 1- (((1-(ssreg/sstot))*(len(y_ln)-1))/(len(y_ln)-1))
    print("R2=",R2)
    
    return -E_IR


cnsts =  leastsq(
            #geneticTL,
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

# this is all just building graphs with the results.


# defines the independent variable. 
#x = np.linspace(data_set[0,0],data_set[-1,0],200)
x = data_set[:,0]

# defines the independent variable. 
#y = np.linspace(data_set[0,0],data_set[-1,0],200)
y =  data_set[:,1]

# read in spectrum from data file
# T=nhietdo, I=cuongdo, dI=I uncertainty

# Create figure window to plot data
fig = plt.figure(1, figsize=(9.5, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#fig, ax1 = plt.subplots(dpi = 300)

#sets the axis labels and parameters.

ax1.tick_params(direction = 'in', pad = 15)
ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)

# plots the first two data sets: the raw data and the GaussSum.
ax1.plot(data_set[:,0], data_set[:,1], 'ko')
ax1.plot(x,GaussSum(x,cnsts, n_value))


# new add
#ax1.fill_between(data_set[:,0], data_set[:,1], facecolor="red", alpha=0.5)

#ax1.fill_between(x, GaussSum(x,cnsts, n_value), facecolor="yellow", alpha=0.25)

# adds a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.plot(
        x, 
        gaussian(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )
    )
# adds color a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.fill_between(
        x, 
        gaussian(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        ),alpha=0.25
    )

'''
# adds a ffGOK of each individual gaussian to the graph.
AE1 = dict()
for i in range(n_value):
    AE1[i] = AE(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )
    
'''
    
# adds a ffGOK of each individual gaussian to the graph.
AE2 = dict()
for i in range(n_value):
    AE2[i] = AE_gen(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )

# adds a ffGOK of each individual gaussian to the graph.
AE3 = dict()
for i in range(n_value):
    AE3[i] = AE_IR(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )

# adds a ffGOK of each individual gaussian to the graph.
GA1 = dict()
for i in range(n_value):
    GA1[i] = geneticTL(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )
# adds a ffGOK of each individual gaussian to the graph.
GA2 = dict()
for i in range(n_value):
    GA2[i] = geneticTLy(
            x, 
            cnsts[4*i], 
            cnsts[4*i+1], 
            cnsts[4*i+2],
            cnsts[4*i+3]
        )

# creates ledger for each graph
ledger = ['Data', 'Resultant']
for i in range(n_value):
    ledger.append(
        'P' + str(i+1)
        #+ ', E = ' + str(round(AE1[i],3)) + ' eV'
        + ', Eg = ' + str(round(AE2[i],3)) + ' eV'
        #+ ', E_IR = ' + str(round(AE3[i],3)) + ' eV'
        #+ '\nTm_gen = ' + str(round(GA1[i],3)) + ' K'
        #+ '\nIm_gen = ' + str(round(GA2[i],3)) + ' a.u.'
    ) 

#adds the ledger to the graph.
ax1.legend(ledger)

#adds text FOM
ax1.text(0.12, 0.65, r'General order kinetic (GOK)''\n FOM = {0:0.8f}'
         .format(abs(FOM(cnsts, y, x, n_value))), transform=ax1.transAxes)

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.plot(x,residuals(cnsts, y, x, n_value))
#ax2.plot(x,GaussSum(x,cnsts, n_value))

ax2.set_xlabel('Temperature (K)',fontsize = 15)
ax2.set_ylabel('Residuals', fontsize = 15)
#ax2.set_ylim(-20, 20)
#ax2.set_yticks((-20, 0, 20))

# format and show
plt.tight_layout()

fig.savefig(r'Rg1_Res_GOK.png')
plt.show()
