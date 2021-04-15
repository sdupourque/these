import pyproffit
import numpy as np
from abel.hansenlaw import hansenlaw_transform

def BetaModel3D(x, beta, rc, norm, bkg):
    return np.power(10., norm) * np.power(1. + (x / rc) ** 2, -3. * beta / 2.)

class ProjectedModel(pyproffit.Model):

    def __init__(self, *args, **kwargs):
        self.maxrad = kwargs.pop('maxrad')
        self._xp = np.linspace(0, 2*self.maxrad, 10000)
        self._dx = self._xp[1] - self._xp[0]
        super(ProjectedModel,self).__init__(*args, **kwargs)

    def __call__(self, x, *pars):

        c2 = 10. ** pars[-1]
        fp = hansenlaw_transform(self.model(self._xp, *pars[:-1], -np.inf), dr=self._dx, direction='forward') + c2

        return np.interp(x, self._xp, fp)

if __name__=='__main__':

    import matplotlib.pyplot as plt

    mod = ProjectedModel(BetaModel3D, maxrad=15.)
    mod.SetParameters([2/3, 3, -2.5, -5])

    r = np.linspace(0, 15, 1000)

    plt.plot(r, mod.model(r, *mod.params))
    plt.plot(r, mod(r, *mod.params))

    plt.loglog()
    plt.show()
