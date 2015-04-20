#
# implementation of the inverse covarince with ADMM method (Alternating Direction Method of Multipliers by Boyd et.al) 
# send commnets about implementation to sergey.plyasunov@gmail.com
#
import sys
import numpy as np
 
def cov(X, assume_centered=False):
    '''empirical covariance based on samples 
    '''
    X = np.asarray(X)
   
    if(X.ndim == 1):
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
    if assume_centered:
        C = np.dot(X.T, X) / X.shape[0]
    else:
        C = np.cov(X.T, bias=1)
    return C
 
 
def shrinkage(x, kappa):
    y = np.maximum(0,  x - kappa) - np.maximum(0, -x - kappa)
    return y
 
def objective(S, X, Z, rho):
    obj = np.matrix.trace(np.dot(S,X)) - np.log( np.linalg.det(X) ) \
          + rho * np.linalg.norm(Z, 1)
    return obj
 
class History:
    def __init__(self):
        '''
        '''
       
        self.objval = []
        self.r_norm = []
        self.s_norm = []
        self.eps_pri = []
        self.eps_dual = []
       
 
        
#@profile
def covsel(D, Lambda, rho, alpha):
    '''ADMM algorithm
    '''
    QUIET    = 0
    MAX_ITER = 100
    ABSTOL   = 1e-4
    RELTOL   = 1e-2
    ###
    S = cov(D)
    #S = 0.5*(S+S.T)
    n = S.shape[0]
    X = np.zeros((n,n))
    Z = np.zeros((n,n))
    U = np.zeros((n,n))
    ##history
    history = History()
 
    if not QUIET:
        print '%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n' %('iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective')
 
    for k in xrange(MAX_ITER):
        #iterate
        (D,V) = np.linalg.eig( rho*(Z - U) - S)
       
        es = D[:]
        xi = (es + np.sqrt(es**2 + 4*rho))/(2.0*rho)
        X = np.dot( V, np.dot(np.diag(xi), V.T) )
 
       
        Z_old = Z[:,:]
        X_hat = alpha * X + (1-alpha) * Z_old
        Z = shrinkage(X_hat + U, Lambda/rho);
        U = U + (X_hat - Z)
       
        #update history#
        history.objval.append( objective(S, X, Z, Lambda) )
        history.r_norm.append( np.linalg.norm(X-Z,'fro') )
        history.s_norm.append( np.linalg.norm(-rho*(Z-Z_old),'fro') )
        history.eps_pri.append(  np.sqrt(n*n)*ABSTOL + RELTOL * np.maximum( np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro') ) )
        history.eps_dual.append( np.sqrt(n*n)*ABSTOL + RELTOL * np.linalg.norm(rho * U,'fro') )
        
        
        if not QUIET:
            print "%d\t%f\t%f\t%f\t%f\t%f"%( k, history.r_norm[-1], history.eps_pri[-1],history.s_norm[-1], history.eps_dual[-1], history.objval[-1])
        
        if ((history.r_norm[-1] < history.eps_pri[-1]) and (history.s_norm[-1] < history.eps_dual[-1])):
            break
    
    return (Z,history)
       
if __name__=="__main__":
   
    import time
    t0 = time.clock()
    D = np.random.normal(size=(100,90))
    C = cov(D)
    t1 = time.clock()
   
    print C.shape
    print t1-t0
 
    (Z,history)=covsel(D,1.0,1.0,1.0)
    #import pylab as pl
    #for i in range(3):
    #    pl.plot(Z[i,:])
    #    pl.plot(C[i,:])
       
    #pl.grid()
    #pl.show()
    print Z
    print C
