from __future__ import division
import numpy as np

def l1tf(y,lambda_,maxiter=40,tol=1E-4):
    # pythonized version of the L1-trend filtering:
    # (x,status) = l1tf(y,lambda,maxiter,tol)
    #
    # finds the solution of the l1 trend estimation problem
    #
    #  minimize_{x} (1/2)||y-x||^2+lambda*||Dx||_1,
    #
    # with variable x, and problem data y and lambda, with lambda >0.
    # D is the second difference matrix, with rows [0... -1 2 -1 ...0]
    #
    # and the dual problem:
    #
    #  minimize    (1/2)||D'*z||^2-y'*D'*z
    #  subject to  norm(z,inf) <= lambda,
    #
    # with variable z.
    #
    # Input arguments:
    #
    # - y:          n-vector; original signal
    # - lambda:     scalar; positive regularization parameter
    #
    # Output arguments:
    #   tuple:
    # - x:          n-vector; primal optimal point
    # - status:     string;
    #               'solved', 'maxiter exceeded'
    #
    # for more details,
    # see "l1 Trend Filtering", S. Kim, K. Koh, ,S. Boyd and D. Gorinevsky
    # www.stanford.edu/~boyd/l1_trend_filtering.html
    #
    #----------------------------------------------------------------------
    #               INITIALIZATION
    #----------------------------------------------------------------------
    
    # PARAMETERS
    ALPHA     = 0.01   # backtracking linesearch parameter (0,0.5]
    BETA      = 0.5    # backtracking linesearch parameter (0,1)
    MU        = 2      # IPM parameter: t update
    MAXITER   = maxiter     # IPM parameter: max iteration of IPM
    MAXLSITER = int(maxiter/2.0)     # IPM parameter: max iteration of line search
    TOL       = tol   # IPM parameter: tolerance
    
    # DIMENSIONS
    n   = len(y)    # length of signal x
    m   = n-2          # length of Dx
    I2  = np.eye(n-2)
    O2  = np.zeros((n-2,1))
    
    #D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
    D = np.zeros((m,n))
    e = np.zeros(n)
    e[0:3] = np.array([1.,-2.,1.])
    for j in range(m): D[j,:]=np.roll(e,j)
        
    DDT = np.dot(D,D.T)
    Dy  = np.dot(D,y)
    
    print Dy.shape

    x = y[:]
    z   = np.zeros(m)   # dual variable
    mu1 = np.ones(m)    # dual of dual variable
    mu2 = np.ones(m)    # dual of dual variable

    t    = 1e-10
    pobj =  np.Inf
    dobj =  0.0
    step =  np.Inf
    f1   =  +z-lambda_
    f2   =  -z-lambda_
    #
    for iters in range(MAXITER):
        DTz  = np.dot(z.T,D).T
        DDTz = np.dot(D,DTz)
        w    = Dy-(mu1-mu2)
        print "w:->",w.shape
        
        pobj1 = 0.5*np.dot(w.T,np.linalg.solve(DDT,w)) + lambda_* np.sum(mu1 + mu2)
        pobj2 = 0.5*np.dot(DTz.T,DTz) + lambda_* np.sum(np.abs(Dy-DDTz))
        pobj  = np.fmin(pobj1,pobj2)
        dobj  = -0.5*np.dot(DTz.T,DTz) + np.dot(Dy.T,z)
        gap   =  pobj - dobj
        # STOPPING CRITERION
        if (gap <= TOL):
            status = 'solved'
            x = y-np.dot(D.T,z)
            print(status)
            return (x,status)
        
        if (step >= 0.2):
            t =max(2.0*m*MU/gap, 1.2*t)
        
        # CALCULATE NEWTON STEP  
        rz      =  DDTz - w
        S       =  DDT-np.diag(mu1/f1+mu2/f2)
        r       = -DDTz + Dy + (1/t)/f1 - (1/t)/f2
        dz      =  np.linalg.solve(S,r)
        dmu1    = -(mu1+((1/t)+dz*mu1)/f1)
        dmu2    = -(mu2+((1/t)-dz*mu2)/f2)

        resDual = rz[:]
        resCent = np.vstack((-mu1*f1-1/t, -mu2*f2-1/t))
        residual= np.vstack((resDual, resCent))
        
        # BACKTRACKING LINESEARCH
        negIdx1 = (dmu1 < 0.0)
        negIdx2 = (dmu2 < 0.0)
        step = 1.0

        if (np.any(negIdx1)):
            step = min( step, 0.99*min(-mu1[negIdx1]/dmu1[negIdx1]) )
        if (np.any(negIdx2)):
            step = min( step, 0.99*min(-mu2[negIdx2]/dmu2[negIdx2]) )
            
        for liter in range(1,MAXLSITER):
            newz    =  z  + step*dz
            newmu1  =  mu1 + step*dmu1
            newmu2  =  mu2 + step*dmu2
            newf1   =  newz - lambda_
            newf2   = -newz - lambda_
    
            # UPDATE RESIDUAL
            newResDual  = np.dot(DDT,newz) - Dy + newmu1 - newmu2
            newResCent  = np.vstack((-newmu1*newf1-1/t, -newmu2*newf2-1/t))
            newResidual = np.vstack((newResDual, newResCent))
            
            if ( (max(max(newf1),max(newf2)) < 0.0) and (np.linalg.norm(newResidual) <= (1.0-ALPHA*step)*np.linalg.norm(residual)) ):
                break
            step = BETA*step
            
        #Update primal and dual variables:
        z  = newz[:]
        mu1 = newmu1[:]
        mu2 = newmu2[:]
        f1 = newf1[:]
        f2 = newf2[:]
        # The solution may be close at this point, but does not meet the stopping
        # criterion (in terms of duality gap).
        x = y-np.dot(D.T,z)

        if (iters >= MAXITER):
            status = 'maxiter exceeded';
            print(status);
            return (x,status)

if __name__=="__main__":
    pass
