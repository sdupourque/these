import numpy as np
import matplotlib.pyplot as plt
from simu.random_field import Fluctuation, pl_cutoff_PowerSpectrum

k = np.geomspace(1e-6, 1e1, 1000)

plt.plot(k , pl_cutoff_PowerSpectrum(k))
plt.ylim(bottom=1e-12, top=1e5)
plt.xlabel('k [kpc-1]')
plt.ylabel('P3D')
plt.loglog()
plt.show()