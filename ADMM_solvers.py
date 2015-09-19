from __future__ import print_function
import numpy as np

def lstsqb_admm(b,A,x0,lb,ub,niter=10):

    rho = 1.01#how to select this?
    m,n = len(b),len(x0)
    x_k= Pr(x0, lb, ub)
    u_k = np.zeros(n)
    z_k = np.zeros(n)
    Atb = np.dot(A.T,b)
    AtA = np.dot(A.T,A) + rho * np.eye(n)
    
    for k in xrange(niter):
        sol = np.linalg.lstsq(AtA, Atb + rho*(z_k - u_k))
        x_k = sol[0]
        z_k = Pr(x_k + u_k, lb,ub)
        u_k = u_k + (x_k - z_k)
        #print(z_k)          
    return (x_k,z_k,u_k)

if __name__=="__main__":
    pass
