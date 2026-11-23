"""Verifica ejemplos con SVD y un optimizador independiente de Newton."""
import numpy as np
from scipy.optimize import minimize
from sections import data as d

expected = np.linalg.lstsq(d.XH, d.HYPM, rcond=None)[0]
assert np.allclose(expected, [d.OLS0,d.OLS1])
assert np.isclose(d.ols_sse(*expected),3.2)
assert len(d.PERC)==4 and d.separates(d.PERC[-1][1],d.PERC[-1][2])
for (_,oldw,oldb),(idx,w,b) in zip(d.PERC,d.PERC[1:]):
    x,y,c=d.TWO[idx]
    assert c*d.g(oldw,oldb,(x,y))<=0
    assert np.allclose(w,np.array(oldw)+d.PERC_ETA*c*np.array([x,y]))
    assert np.isclose(b,oldb+d.PERC_ETA*c)

def obj(beta):
    z=d.XH@beta
    return np.mean(np.logaddexp(0,z)-d.HY01*z)

fit=minimize(obj,np.zeros(2),method='BFGS',tol=1e-9)
assert np.allclose(fit.x,d.LOG_BETA,atol=1e-5)
assert np.linalg.matrix_rank(d.XH)==2
assert np.linalg.norm(d.XH.T@(d.LOG_P-d.HY01))<1e-8
assert np.isclose(d.LOG_CUT,3.5)
assert d.J_FISHER>=d.J_CURVE.max()-1e-10
assert np.all(np.linalg.eigvalsh(d.S_W)>0)
assert np.allclose(d.S_W@d.W_FISHER,d.M_A-d.M_B)
assert [d.n_errors(b,p) for b,p in [(d.MASK_OLS,d.MASK_BASE),(d.MASK_OLS_FAR,d.MASK_ALL),
    (d.MASK_LOG,d.MASK_BASE),(d.MASK_LOG_FAR,d.MASK_ALL)]]==[0,3,0,0]
for (k,j),v in d.SOFT_RAYS:
    scores=d.soft_scores(*(d.TRIPLE+v))
    assert np.isclose(scores[k],scores[j]) and scores[k]>=max(scores)-1e-10
assert np.isclose(d.phi_quad(d.KQ_X)@d.phi_quad(d.KQ_Z),25)
assert np.linalg.eigvalsh(d.K_LINE).min()>-1e-10
assert np.allclose((d.K_LINE+d.RBF_LAMBDA*np.eye(len(d.LX)))@d.RBF_ALPHA,d.LY)
assert np.all(np.sign(d.rbf_f(d.LX))==d.LY)
assert [z for _,_,z,_ in d.EX_ROWS]==[2,-3,0]
print('Verificados: MCO/SVD, perceptrón, logística/BFGS, Fisher, softmax, kernels y ejercicio.')
