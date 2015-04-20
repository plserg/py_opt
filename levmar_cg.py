#
# Hessian free version of the LevMar method
# send comments to sergey.plyasunov@gmail.com
#

import numpy as np
from collections import namedtuple
import warnings

Result = namedtuple('Result',['xopt','gopt','n_calls'])

def levmar_cg(func_grad_hess,x0, args=(),tol=1e-4,maxiter=100,callback=None):
    """
    Minimization of scalar function of one or more variables using the
    Newton-LevMar-CG algorithm.

    Parameters
    ----------
    func_grad_hess : callable
        Should return the value of the function, the gradient, and a
        callable returning the matvec product of the Hessian.

    x0 : float
        Initial guess.

    args: tuple, optional
        Arguments passed to func_grad_hess, func and grad.

    tol : float
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int
        Number of iterations.

    Returns
    -------
    namedtuple(xopt,gopt,n_calls)
    """
    x0 = np.asarray(x0).flatten()
    xk = x0
    (fval, fgrad, fhess_p) = func_grad_hess(xk, *args)
    k = 1
    n_calls =1
    lm_gain_thld = 0.25
    nu = 2.0
    mu = 1e-4 * np.max(np.abs(np.diag(fhess_p)))

    # Outer loop: our Newton iteration
    while k <= maxiter:
        # Compute a search direction pk by applying the CG method to
        #  H p = - fgrad f(xk) starting from 0.
        (fval, fgrad, fhess_p) = func_grad_hess(xk, *args)
        n_calls +=1

        absgrad = np.abs(fgrad)
        if np.max(absgrad) < tol:
            break

        grad_norm = np.sum(absgrad)
        eta = min([0.5, np.sqrt(grad_norm)])
        termcond = eta * grad_norm
        #descent direction
        dc = np.zeros(len(x0), dtype=x0.dtype)
        r_i = fgrad
        psup_i = -r_i
        i = 0
        rho0 = np.dot(r_i, r_i)
        cg_maxiter = maxiter
        # Inner loop: 
        # solve the LM-Newton update by conjugate gradient (CG), to
        # avoid inverting the Hessian
        while np.sum(np.abs(r_i)) > termcond or (i<=cg_maxiter):
            Hp = np.dot(fhess_p,psup_i) + mu * psup_i##LM-step
            # check curvature
            curv = np.dot(psup_i, Hp)
            if( 0 <= curv <= 3.0 * np.finfo(np.float64).eps):
                break
            elif curv < 0.0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    dc = dc + (rho0 / curv) * psup_i
                    break
            alpha_i = rho0 / curv
            dc = dc + alpha_i * psup_i
            r_i = r_i + alpha_i * Hp
            rho1 = np.dot(r_i, r_i)
            beta_i = rho1 / rho0
            psup_i = -r_i + beta_i * psup_i
            rho0 = rho1          # update np.dot(ri,ri) for next time.
            i = i+1

        ##now we are supposed to have dc: H * dc= -grad
        xt = xk + dc
        (fval_t, fgrad_t, fhess_t) = func_grad_hess(xt, *args)
        n_calls +=1
        lm_gain = 2*(fval -fval_t)/np.dot(dc,mu*dc-fgrad)
        if(lm_gain>lm_gain_thld):
            xk = xt[:]
            mu = mu * max(0.333, 1.0-(2.0*lm_gain-1)**3)
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2.0 * nu

        if callback is not None:
            callback(xk)
        k =k + 1


    if k > maxiter:
        warnings.warn("newton-cg failed to converge. Increase the number of iterations.")
    return Result(xopt=xk,gopt=fgrad,n_calls=n_calls)


from scipy import optimize

def func_grad_hess(x,*args):

    f = optimize.rosen(x)
    g = optimize.rosen_der(x)
    h= optimize.rosen_hess(x)
    return (f,g,h)

ndim=100


x0=0.415*np.random.normal(size=ndim)

res=levmar_cg(func_grad_hess,x0,tol=1e-4,maxiter=1000)

x = res.xopt
g = res.gopt

print x
print np.fabs(g)
print res.n_calls

