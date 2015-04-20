from cvxopt import matrix,spmatrix,spdiag
from cvxopt import solvers
from cvxopt import lapack
 
def L1Regression(A,b,alpha,trace=False):
   """
    solves
    minimize_{x} ||A*x-b||^2_2 + alpha*||x||_1
   """
   (n,m)=A.size
   Id = spmatrix(1.0, range(m),range(m)) ##unit matrix
   G = matrix([A.T,-A.T])
   h = matrix(alpha, (2*m,1))
   solvers.options['show_progress'] = trace
   sol= solvers.qp(Id, b, G, h)
   z= sol['x']
   x= A.trans()*(b-z)
   A_tA = A.trans()*A
   lapack.gesv(A_tA, x)
   return x 
 
def L1RegressionQP(A,b,W,alpha,trace=False):
   '''
   minimize_{x} ||A*x-b||^2_2 + alpha*||W*x||_1
   '''
   solvers.options['show_progress'] = trace
   (n,m) = A.size
   Id = spmatrix(1.0, range(m),range(m))
   P  = spdiag([A.T*A,Id])
   q  = matrix( [-A.T*b, alpha*matrix(1.0,(m,1))],(2*m,1))
   #print(P)
   #print(e)
   #q  = alpha * e
   G =  matrix([[W,-Id],[-W,-Id]])
   h = matrix(0.0,(2*m,1))
   sol=solvers.qp(P, q, G, h)
   x = sol['x']
   return x[0:m]
  
if __name__=="__main__":
  
   A = matrix([ [20000, .5, 0], [.5, 100,0],[0 ,0 ,1000.0] ])
   b = matrix([1.0,1.0,1.0])
   alpha =0.0001
   (n,m) = A.size
   W =  spmatrix(1.0, range(m),range(m))
  
   ##solve
   x= L1Regression(A,b,alpha)
   print(x)
   print("err="); print(A*x-b)
   x= L1RegressionQP(A,b,W,alpha)
   print(x)
   print("err="); print(A*x-b)
