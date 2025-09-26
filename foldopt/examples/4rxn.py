from foldopt.models import IrbackModel
from foldopt.opt import OptAlgo

import matplotlib.pyplot as plt
import numpy as np

x = IrbackModel(sequence="MKKYTCTVCGYIYDPEDGDPDDGVNPGTDFKDIPDDWVCPLCGVGKDEFEEVEE")
x.set_optimizer(OptAlgo.METROPOLIS_ANNEALING)
out = x.optimize(initial_temp=100.0, final_temp=1e-12, cooling_rate=0.9999, lam = 3.0)

print(f'Optimal energy: {out.optimal_energy}')
np.savez('4rxn_run.npz', energies=out.energies, accepts=out.accepts, rejects=out.rejects, conformations=[c.conformation for c in out.conformations])
plt.plot(out.energies[1000:])
plt.show()
plt.close() 

plt.plot(np.cumsum(out.accepts))
plt.show()