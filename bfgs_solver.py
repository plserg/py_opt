# Copyright (c) 2013, sergey.plyasunov@gmail.com
# *computes Hessian and inverse Hessian BFGS style 
# *added barzilai-borwein gradient descent

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
from collections import namedtuple

Result = namedtuple('Result', ['xopt', 'gopt','H' ,'invH', 'n_grad_calls'])

def fmin_bfgs(fun, fprime, x0, args=(), gtol=1e-5,  H0=None, B0=None, callback=None, maxiter=None):

    """Minimize a function, via only information about its gradient, using BFGS

    The difference between this and the "standard" BFGS algorithm is that the
    line search component uses a weaker criterion, because it can't check
    for sure that the function value actually decreased.

    Parameters
    ----------
    fun : callable f(x, *args)
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    x0 : ndarray
        Initial guess
    args : tuple, optional
        Extra arguments to be passed to `fprime`
    H0 : initial Hessian
    B0 : inverse Hessian
    gtol : float, optional
        gradient norm must be less than `gtol` before succesful termination
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as `callback(xk)`, where `xk` is the current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize `f`, and which are a root of the gradient,
        `fprime`
    gopt : ndrarray
        value of the gradient at `xopt`, which should be near zero
    Hopt : ndarray
        final estimate of the hessian matrix at `xopt`
    n_grad_calls : int
        number of gradient calls made
    """
    c1 = 0.1e0
    beta = 0.99e0
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
            maxiter = int(len(x0)*0.5)

    gf_k = fprime(x0, *args)  # initial gradient
    gf_kp1 = gf_k.copy()
    n_grad_calls = 1  # number of calls to fprime()

    k = 0  # iteration counter
    n = len(x0)  # degreees of freedom
    I = np.eye(n, dtype=np.float32)

    if B0 is not None:
        B_k = B0.copy()  # initial guess of the inverse Hessian Hk=Bk^{-1}
    else:
        B_k =I.copy()
        B0 = B_k.copy()

    if H0 is not None:
        H_k = H0.copy()
    else:
        H_k = I.copy()
        H0= H_k.copy()

    b0 = np.diag(B_k)
    x_k = x0.copy()
    s_k = np.zeros(n)
    y_k = np.zeros(n)
    

    s_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    y_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    rho_store = np.array([],dtype=np.float32)
    #print "--->",lbfgs_hessian_solve(gfk,b0,s_store,y_store,rho_store)-np.dot(Bk,gfk)
    gnorm = np.linalg.norm(gf_k)

    while (gnorm > gtol) and (k < maxiter):
        # search direction
        d_k = -np.dot(B_k, gf_k)
        #print "--->",lbfgs_prod_vec(gfk,b0,s_store,y_store,rho_store)-np.dot(Bk,gfk)
        gd = np.dot(gf_k.T,d_k)
        dHd = np.dot(d_k.T,np.dot(H_k,d_k))
        alpha_k = -gd/dHd

        f_tgt = fun(x_k+alpha_k*d_k,*args)
        f_k = fun(x_k,*args)

        while (f_tgt> f_k+ c1*alpha_k*(gd + 0.5*alpha_k*dHd)):
            alpha_k *=beta
            f_tgt = fun(x_k+alpha_k*d_k,*args)

        #(alpha_k, gfkp1, ls_grad_calls) = _line_search(fprime, x_k, gf_k, d_k, args)
        #n_grad_calls += ls_grad_calls
        
        # advance in the direction of the step
        x_kp1 = x_k + alpha_k * d_k
        s_k = x_kp1 - x_k
        x_k = x_kp1.copy()

        gf_kp1 = fprime(x_kp1, *args)
        n_grad_calls += 1

        y_k = gf_kp1 - gf_k
        gf_k = gf_kp1.copy()

        if callback is not None:
            callback(x_k)

        k += 1
        gnorm = np.linalg.norm(gf_k)
        if( gnorm < gtol ):
            break
        
        if np.dot(y_k.T,s_k)>0.0e0:

            rho_k = 1.0e0 /np.dot(y_k.T, s_k)
            rho_store = np.append(rho_store,rho_k)
            s_store = np.append(s_store,s_k.reshape(n,1),1)
            y_store = np.append(y_store,y_k.reshape(n,1),1)

            V_H = I -  rho_k * np.outer(s_k[:],y_k[:].T)
            V_B = I -  rho_k * np.outer(y_k[:],s_k[:].T)
            H_k = np.dot(V_H.T, np.dot(H_k, V_H)) + rho_k * np.outer(y_k[:],y_k[:].T)
            B_k = np.dot(V_B.T, np.dot(B_k, V_B)) + rho_k * np.outer(s_k[:],s_k[:].T)
        else:
            H_k=H0
            B_k=B0

    if k >= maxiter:
        print "Warning: %d iterations exceeded" % maxiter
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k


    elif gnorm < gtol:
        print "Optimization terminated successfully."
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k

    return Result(xopt=x_k, gopt=gf_k, invH=B_k, H=H_k ,n_grad_calls=n_grad_calls)

def fmin_pbfgs(fun, fprime, x0, args=(), gtol=1e-5,  H0=None, B0=None, bc=None, callback=None, maxiter=None):
   
    """Minimize a function, via only information about its gradient, using projected  BFGS
  
    """
    x0 = np.asarray(x0).flatten()
    if maxiter is None:
            maxiter = int(len(x0)*0.5)
   
    gf_k = fprime(x0, *args)  # initial gradient
    gf_kp1 = gf_k.copy()
    n_grad_calls = 1  # number of calls to fprime()
   
    
    k = 0  # iteration counter
    n = len(x0)  # degreees of freedom
    I = np.eye(n, dtype=np.float32)
   
    B_k = B0.copy()  # initial guess of the inverse Hessian Hk=Bk^{-1}
    H_k = H0.copy()
    b0 = np.diag(B0)
   
    x_k = x0[:]
    s_k = np.zeros(n)
    y_k = np.zeros(n)
    alist = np.zeros(n,dtype=np.int)
   
    pgf_k = x_k - Proj(x_k-gf_k, bc)
    eps = 1e-1##boundary detection
   
    s_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    y_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    rho_store = np.array([],dtype=np.float32)
    #print "--->",lbfgs_hessian_solve(gfk,b0,s_store,y_store,rho_store)-np.dot(Bk,gfk)
    gnorm = np.linalg.norm(gf_k)
    pgnorm = np.linalg.norm(pgf_k)
   
    ##update active set list
    if bc is not None:
        up_m_low = np.array([bc[i][1]-bc[i][0] for i in xrange(n)])
        up_low_min = np.min( up_m_low )
        eps = 1e-4*min(pgnorm,0.5*up_low_min)     
        for i in xrange(n):
            if( (bc[i][0]<x_k[i] and  (bc[i][0]+eps)>x_k[i] and gf_k[i]>0.0e0) or (bc[i][1]>x_k[i] and  (bc[i][1]-eps)<x_k[i] and gf_k[i]<0.0e0)):
                alist[i]=1
  
   
    while (pgnorm > gtol) and (k < maxiter):
        # search direction
        p_k = -np.dot(B_k, gf_k)
        #print "--->",lbfgs_prod_vec(gf_k,b0,s_store,y_store,rho_store)-np.dot(B_k,gf_k)
        p_k = p_k + Proj_A(-gf_k,alist)
        (alpha_k, gf_kp1, ls_grad_calls) = _line_search(fprime, x_k, gf_k, p_k, args, alpha_guess=1.0,curvature_condition=0.9,update_rate=0.5, maxiters=10)     
        n_grad_calls += ls_grad_calls
       
        # advance in the direction of the step
        x_kp1 = Proj(x_k + alpha_k * p_k, bc)
        s_k = x_kp1 - x_k
        x_k = x_kp1.copy()
       
        #if gfkp1 is None:
        gf_kp1 = fprime(x_kp1, *args)
        n_grad_calls += 1
       
        y_k = gf_kp1 - gf_k
        gf_k = gf_kp1[:]
       
        pgf_k = x_k - Proj(x_k-gf_k,bc)
        gnorm = np.linalg.norm(gf_k)
        pgnorm = np.linalg.norm(pgf_k)
       
        #project displacement and gradient
        s_k = Proj_I(s_k,alist)
        y_k = Proj_I(y_k,alist)
       
        if callback is not None:
            callback(x_k)
 
        ##update active set list
        alist = np.zeros(n,dtype=np.int)
        if bc is not None:
            up_m_low = np.array([bc[i][1]-bc[i][0] for i in xrange(n)])
            up_low_min = np.min( up_m_low )
            eps = 1e-4*min(pgnorm,0.5e0*up_low_min)
           
            for i in xrange(n):
                   if((bc[i][0]<x_k[i] and (bc[i][0]+eps)>x_k[i] and gf_k[i]>0.0e0) or (bc[i][1]>x_k[i] and  (bc[i][1]-eps)<x_k[i] and gf_k[i]<0.0e0)):
                          alist[i]=1
                   
        k += 1  
      
        ##check the curvetaure conditions
        if np.dot(y_k.T,s_k)>0.0e0:
            rho_k = 1.0e0 /np.dot(y_k.T, s_k)
            rho_store = np.append(rho_store,rho_k)
            s_store = np.append(s_store,s_k.reshape(n,1),1)
            y_store = np.append(y_store,y_k.reshape(n,1),1)
       
            V_H = I -  rho_k * np.outer(s_k[:],y_k[:].T)
            V_B = I -  rho_k * np.outer(y_k[:],s_k[:].T)
            H_k = np.dot(V_H.T, np.dot(H_k, V_H)) + rho_k * np.outer(y_k[:],y_k[:].T)
            B_k = np.dot(V_B.T, np.dot(B_k, V_B)) + rho_k * np.outer(s_k[:],s_k[:].T)
        else:
            print "curveture break-down np.dot(y_k.T,s_k)=%f...restaring :"%(np.dot(y_k.T,s_k))
            B_k = B0.copy()  # initial guess of the inverse Hessian Hk=Bk^{-1}
            H_k = H0.copy()
       
    
    if k >= maxiter:
        print "Warning: %d iterations exceeded" % maxiter
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
       
    
    elif pgnorm < gtol:
        print "Optimization terminated successfully."
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k
     
   
    return Result(xopt=x_k, gopt=gf_k, invH=B_k, H=H_k ,n_grad_calls=n_grad_calls)
 

def fmin_grad_descent(fun, fprime, x0, args=(), gtol=1e-5,H0=None,B0=None, bc=None,  callback=None, maxiter=None):

    """Minimize a function,
    via only information about its gradient, using BFGS combined with
Barzilai-Borwain method
    There is no line search but we still compute Hessian via BFGS like step
    Parameters
    ----------
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    x0 : ndarray
        Initial guess
    args : tuple, optional
        Extra arguments to be passed to `fprime`
    gtol : float, optional
        gradient norm must be less than `gtol` before succesful termination
    callback : callable, optional
        An optional user-supplied function to call after each iteration.
        Called as `callback(xk)`, where `xk` is the current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize `f`, and which are a root of the gradient,
        `fprime`
    gopt : ndrarray
        value of the gradient at `xopt`, which should be near zero
    Hopt : ndarray
        final estimate of the hessian matrix at `xopt`
    n_grad_calls : int
        number of gradient calls made
    """
    alpha_min =1e-5
    alpha_max = 1e+5
    nu =1e-4
    alpha_bb=1.0e0
    alpha_k=1.0e0
    h = 10

    x0 = np.asarray(x0).flatten()

    if maxiter is None:
            maxiter = int(len(x0)*0.9)

    gfk = fprime(x0, *args)  # initial gradient

    gfkp1 = gfk[:]
    n_grad_calls = 1  # number of calls to fprime()

    k = 0  # iteration counter
    n = len(x0)  # degreees of freedom
    I = np.eye(n, dtype=np.float32)
    # initial guess of the inverse Hessian Hk=Bk^{-1}
    if B0 is None:
        Bk = I.copy()
    else:
        Bk= B0.copy()
    if H0 is None:
        Hk = I.copy()
    else:
        Hk = H0.copy()

    b0 = 1.0/np.diag(Hk)

    xk = x0[:]
    sk = np.zeros(n)
    yk = np.zeros(n)
    pk = proj(x0-alpha_k*gfk,bc)-x0

    f_store =np.array([],dtype=np.float32,ndmin=2).reshape(1,0) #store the function values
    s_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    y_store=np.array([],dtype=np.float32,ndmin=2).reshape(x0.shape[0],0)
    alpha_store = np.array([],dtype=np.float32)
    rho_store = np.array([],dtype=np.float32)
    gnorm = np.linalg.norm(gfk)

    pgnorm = np.linalg.norm(pk)
    alpha_bb = 1.0/gnorm

    while (pgnorm > gtol) and (k < maxiter):

        alpha_k = min(  alpha_max, max(alpha_min,alpha_bb) )
        alpha_store = np.append(alpha_store,alpha_k)
        f_store= np.append(f_store,fun(xk,*args))
        # search direction
        pk = proj(xk-alpha_k*gfk,bc)-xk
        pgnorm = np.linalg.norm(pk)
        # search direction
        #pk = -gfk
        gx = np.dot(gfk,pk)
        xHx = np.dot(pk.T,np.dot(Hk,pk) )
        f_b = np.max(f_store[0:min(len(f_store),h)])

        alpha=1.0e0
        q_k_alpha = f_store[-1] + alpha*gx+0.5*xHx*(alpha*alpha)
        ##apply Wolfe condition search:
        while(q_k_alpha > (f_b + nu*alpha*gx)):
            alpha = 0.25*alpha
            q_k_alpha = f_store[-1] + alpha*gx+0.5*xHx*(alpha*alpha)

        # advance in the direction of the step
        xkp1 =xk + alpha * pk
        sk = xkp1 - xk
        xk = xkp1[:]
        #
        gfkp1 = fprime(xkp1, *args)
        n_grad_calls += 1
        yk = gfkp1 - gfk
        gfk = gfkp1[:]
        #
        alpha_bb = np.dot(sk.T,yk)/np.dot(yk.T,yk)

        if callback is not None:
            callback(xk)

        rho_k = 1.0e0 /np.dot(yk.T, sk)
        ##store the composition of the inverse Hessian:
        rho_store = np.append(rho_store,rho_k)
        s_store   = np.append(s_store,sk.reshape(n,1),1)
        y_store   = np.append(y_store,yk.reshape(n,1),1)
        ##BFGS-like upadte step
        V_H = I -  rho_k * np.outer(sk[:],yk[:].T)
        V_B = I -  rho_k * np.outer(yk[:],sk[:].T)

        Hk = np.dot(V_H.T, np.dot(Hk, V_H)) + rho_k * np.outer(yk[:],yk[:].T)
        Bk = np.dot(V_B.T, np.dot(Bk, V_B)) + rho_k * np.outer(sk[:],sk[:].T)

        k = k+1

    if k >= maxiter:
        print "Warning: %d iterations exceeded" % maxiter
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k


    elif gnorm < gtol:
        print "Optimization terminated successfully."
        print "         Current gnorm: %f" % gnorm
        print "         grad calls: %d" % n_grad_calls
        print "         iterations: %d" % k

    return Result(xopt=xk, gopt=gfk,invH=Bk, H=Hk ,n_grad_calls=n_grad_calls)


def _line_search(fprime, xk, gk, pk, args=(), alpha_guess=1.0,curvature_condition=0.9,update_rate=0.5, maxiters=10):
    """Inexact line search with only the function gradient

    The step size is found only to satisfy the strong curvature condition, (i.e
    the second of the strong Wolfe conditions)

    Parameters
    ----------
    fprime : callable f(x, *args)
        gradient of the objective function to be minimized
    xk : ndarray
        current value of x
    gk : ndarray
        current value of the gradient
    pk : ndarray
        search direction
    args : tuple, optional
        Extra arguments to be passed to `fprime`

    Returns
    -------
    alpha : float
        The step length
    n_evaluations : int
        The number of times fprime() was evaluated
    gk : ndarray
        The gradient value at the alpha, `fprime(xk + alpha*pk)`

    Other Parameters
    -----------------
    alpha_guess : float, default = 1.0
        initial guess for the step size
    curvature_condition : float, default = 0.9
        strength of the curvature condition. this is the c_2 on the wikipedia
        http://en.wikipedia.org/wiki/Wolfe_conditions, and is recommended to be
        0.9 according to that page. It should be between 0 and 1.
    update_rate : float, default=0.5
        Basically, we keep decreasing the alpha by multiplying by this constant
        every iteration. It should be between 0 and 1.
    maxiters : int, default=5
        Maximum number of iterations. The goal is that the line search step is
        really quick, so we don't want to spend too much time fiddling
with alpha
    """
    alpha = alpha_guess
    initial_slope = np.dot(gk, pk)

    for j in xrange(maxiters):
        gk = fprime(xk + alpha * pk, *args)
        if np.abs(np.dot(gk, pk)) < np.abs(curvature_condition * initial_slope):
            break
        else:
            alpha *= update_rate

    # j+1 is the final number of calls to fprime()
    return (alpha, gk, j+1)

def lbfgs_prod_vec(g,h0,s_store,y_store,rho_store):
    '''computes products of the vector and inverse Hessian (B) with the vector x=B*g using stored state and gradient updates and ystore
    '''
    n = g.shape[0]
    assert(s_store.shape[0]==y_store.shape[0] and s_store.shape[1]==y_store.shape[1] and n == s_store.shape[0])

    m= s_store.shape[1]
    alpha = np.zeros(m)
    q = g.copy()
    r = np.zeros(n)

    for i in xrange(m-1,-1,-1):
        alpha[i] = rho_store[i] * np.dot(s_store[:,i].T,q)
        q =q - alpha[i] * y_store[:,i]

    for i in xrange(n):
        r[i] = h0[i] * q[i]

    for i in xrange(0,m):
        beta =rho_store[i] * np.dot(y_store[:,i].T,r)
        r = r + (alpha[i]-beta)*s_store[:,i]

    return r

def Proj(x,bounds=None):
    '''
    performs Projection of the gradient on the boundary conditions
    '''
    n = len(x)
    if bounds is None:
        return x
    else:
        return np.array([ max(bounds[i][0],min(x[i],bounds[i][1]))  for i in range(n)  ])
 
##projection to the active ste
def Proj_A(x,alist):
    n = len(x)
    px =x.copy()
    for i in xrange(n):
        if(alist[i]==0):
            px[i]=0.0e0
    return px
##projection onto the inactive set
def Proj_I(x,alist):
    n = len(x)
    px =x.copy()
    for i in xrange(n):
        if(alist[i]==1):
            px[i]=0.0e0
           
    return px

def fun1(x):
    F = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[1,0,0]])
    K = np.array([1., 0.3, 0.5])
    log_pdot = np.dot(F, x)
    logZ = np.log(sum(np.exp(log_pdot)))
    f = logZ - np.dot(K, x)
    return f

def grad1(x):
    F = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[1,0,0]])
    K = np.array([1., 0.3, 0.5])
    log_pdot = np.dot(F, x)
    logZ = np.log(np.sum(np.exp(log_pdot)))
    p = np.exp(log_pdot - logZ)
    return np.dot(F.T,p)-K

if __name__ == '__main__':

    def callback(x):
        pass

    n= 20
    A = 1000.0*np.eye(n,dtype=np.float32)
    #A[0,1] = 0.000;
    #A[10,1] =-0.510;
    A[0,0] = 0.0000001;
    #A[2,2]=  500.1
    A = 0.5e0 *(A + A.T)
    H0 = np.diag(np.diag(A))
    B0 = np.diag(1.0/np.diag(A))

    b = np.random.normal(0,1.0,n)
    #b = np.ones(n)
    bc = [(-0.5e10,0.9e+10) for i in range(len(b))]
    
    grad = (lambda x: np.dot(A,x)-b)
    fun = (lambda x: 0.5*np.dot(x.T,np.dot(A,x))-np.dot(b.T,x))
    x0 =np.random.normal(0.0,1.0,n)

    result = fmin_bfgs(fun, grad, x0, args=(), gtol=1e-5, H0=H0,B0=B0, callback=callback, maxiter=10)
    #result = fmin_pbfgs(fun, grad, x0, args=(), gtol=1e-5, H0=H0,B0=B0,bc=bc, callback=callback, maxiter=10)
    #result = fmin_grad_descent(fun, grad, x0, args=(), gtol=1e-6,H0=H0, B0=B0, bc=bc, callback=callback, maxiter=30)

    print result.xopt
    print result.gopt,"grad_norm=%f"%(np.linalg.norm(result.gopt))

    print np.diag(np.dot(result.H,result.invH))
    #print np.diag(A-result.H)

    import matplotlib.pyplot as plt
    plt.pcolor(result.H)
    plt.show()

    x0 = np.random.random(3)*10
    result = fmin_bfgs(fun1, grad1, x0, args=(), gtol=1e-4, maxiter=10)
    print result.xopt
    print result.H
    print np.diag(np.dot(result.H,result.invH))

    #x0 = np.random.random(15)*0.01
    #from scipy import optimize
    #result = fmin_bfgs(optimize.rosen, optimize.rosen_der , x0, args=(), gtol=1e-6, maxiter=100)
    #print result.xopt
    #print np.diag(np.dot(result.H,result.invH))
