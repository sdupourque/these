import pyproffit
import numpy as np
from scipy.special import gamma
from abel.hansenlaw import hansenlaw_transform

def BetaModel3D(x, beta, rc, norm, bkg):
    """see Clerc & al 2019 Appendix B"""
    return (10.**norm) * np.power(1 + (x / rc) ** 2, -3*beta)

def BetaModel2D(x, beta, rc, norm, bkg):
    """see Clerc & al 2019 Appendix B"""
    µc = np.sqrt(np.pi)*gamma(3*beta - 1/2)/gamma(3*beta)
    return (10.**norm) * rc * µc * np.power(1 + (x / rc) ** 2, -3*beta + 1/2) + (10.**bkg)

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

class BetaModel(pyproffit.Model):

    def __init__(self):

        self.npar = 4
        self.parnames = ['beta', 'rc', 'norm', 'bkg']
        self.params = None
        self.model3D = BetaModel3D

    def __call__(self, x, *pars):

        return BetaModel2D(x, *pars)

if __name__=='__main__':

    import matplotlib.pyplot as plt

    mod1 = ProjectedModel(BetaModel3D, maxrad=15.)
    mod2 = BetaModel()
    mod1.SetParameters([2/3, 3, -2.5, -5])
    mod2.SetParameters([2 / 3, 3, -2.5, -5])

    r = np.linspace(0, 15, 1000)

    plt.plot(r, np.abs(mod1.model(r, *mod1.params)-mod2.model3D(r, *mod2.params)))
    plt.plot(r, np.abs(mod1(r, *mod1.params)-mod2(r, *mod2.params)))


    plt.loglog()
    plt.show()
