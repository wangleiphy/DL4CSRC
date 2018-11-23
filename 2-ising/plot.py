from config import * 
import numpy as np 
import matplotlib.pyplot as plt

# S = lnZ + beta*<E>
K_exact, lnZ_exact, E_exact, Cv_exact = np.loadtxt('exact.dat', unpack=True)
K_trg, lnZ_trg, E_trg, Cv_trg = np.loadtxt('trg.dat', unpack=True)

fig = plt.figure(figsize=(8, 4))
plt.plot(K_trg[::3], K_trg[::3]*E_trg[::3]+lnZ_trg[::3], 'o', label='TRG+Autograd', lw=2, zorder=9) 
plt.plot(K_exact, K_exact*E_exact+lnZ_exact, label='Onsager', lw=2)
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel(r'Entropy per site')
#plt.axhline(y=0.30647) #critical entropy
#plt.ylim([0.5, 3])

plt.subplots_adjust(bottom=0.2)

#plt.show()
plt.savefig('trg.pdf')
