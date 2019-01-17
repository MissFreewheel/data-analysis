import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

#read dataset and plot it
f=open('dataset.txt','r')
x, y, err = [], [], []
for line in f.readlines()[1:]:
    x.append(float(line.split()[0]))
    y.append(float(line.split()[1]))
    err.append(float(line.split()[2]))

fig, ax = plt.subplots()
plt.errorbar(x, y, yerr=err, fmt='o', capsize=4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('parameters estimation')
#plt.show()

#fit model y=ax + b to dataset
par = np.polyfit(x,y,deg=1)
z = (np.polyval(par, x) - y)/err
chi_2 = np.sum(z ** 2) # = 4.28

f_fit=np.poly1d(par)
plt.plot(x, f_fit(x))
#plt.show()

#calculate the probability
df = len(x) - 2  #degree of freedom = 8
rv = chi2(df)
prob = rv.cdf(chi_2)
#print(1-prob) #0.83

#Determine chi-squared for a grid of parameters
nbins = 100
aa = np.linspace(-0.1, 0.4, nbins)
bb = np.linspace(-0.5, 2.5, nbins)
#AA, BB = np.meshgrid(aa, bb, sparse=True)
#fig1, ax1 = plt.subplots()
#plt.scatter(AA, BB, marker='o')
a_coords, b_coords = np.meshgrid(aa, bb, indexing='ij')
coordinate_grid = np.array([a_coords, b_coords])
#print(coordinate_grid.shape)
CHI_2=np.zeros((nbins,nbins),dtype=float)
levs = [chi_2, chi_2+2.3, chi_2+6.17, chi_2+11.8]
for i in range(nbins):
    for j in range(nbins):
        ZZ = (np.polyval(coordinate_grid[:,i,j],x)-y)/err
        CHI_2[i,j] = np.sum(ZZ ** 2)

fig1, ax1 = plt.subplots()
cs = ax1.contour(a_coords, b_coords, CHI_2, levels=levs)
ax1.clabel(cs, inline=1, fontsize=10)
plt.show()

