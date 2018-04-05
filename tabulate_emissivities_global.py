import numpy as np

data = np.loadtxt('rtg2_data.txt')
print data.shape

z = data[:,0]

e1450_18 = np.log10(data[:,1])
e1450_18_up = np.log10(data[:,3])
e1450_18_low = np.log10(data[:,2])

de1450_18_plus = e1450_18_up - e1450_18
de1450_18_minus = e1450_18 - e1450_18_low

e1450_21 = np.log10(data[:,4])
e1450_21_up = np.log10(data[:,6])
e1450_21_low = np.log10(data[:,5])

de1450_21_plus = e1450_21_up - e1450_21
de1450_21_minus = e1450_21 - e1450_21_low

e912_18 = np.log10(data[:,7])
e912_18_up = np.log10(data[:,9])
e912_18_low = np.log10(data[:,8])

de912_18_plus = e912_18_up - e912_18
de912_18_minus = e912_18 - e912_18_low

e912_21 = np.log10(data[:,10])
e912_21_up = np.log10(data[:,12])
e912_21_low = np.log10(data[:,11])

de912_21_plus = e912_21_up - e912_21
de912_21_minus = e912_21 - e912_21_low

g_18 = np.log10(data[:,13])
g_18_up = np.log10(data[:,15])
g_18_low = np.log10(data[:,14])

dg_18_plus = g_18_up - g_18
dg_18_minus = g_18 - g_18_low

g_21 = np.log10(data[:,16])
g_21_up = np.log10(data[:,18])
g_21_low = np.log10(data[:,17])

dg_21_plus = g_21_up - g_21
dg_21_minus = g_21 - g_21_low

fs = (r'{:.1f} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} \\')

for i in range(len(z)-1,-1,-1):

    s = fs.format(z[i],
                  e1450_18[i], de1450_18_plus[i], de1450_18_minus[i],
                  e1450_21[i], de1450_21_plus[i], de1450_21_minus[i],
                  e912_18[i], de912_18_plus[i], de912_18_minus[i],
                  e912_21[i], de912_21_plus[i], de912_21_minus[i],
                  g_18[i], dg_18_plus[i], dg_18_minus[i],
                  g_21[i], dg_21_plus[i], dg_21_minus[i])

                  
    print s 
