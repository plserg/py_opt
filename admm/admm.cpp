//
//  admm.cpp
//  armadillo_test
//
//  Created by Sergey Plyasunov on 10/5/15.
//  Copyright (c) 2015 Sergey Plyasunov. All rights reserved.
//

#include <stdio.h>
#include <numeric>
#include "admm.h"

namespace admm {
//
// solves argmin{||b-Ax||^2_2}
//s.t. lb<=x<=ub
//
int qp(arma::vec& xopt,/*solution out*/
             const arma::mat& A,
             const arma::vec& b,
             const arma::vec& lb,
             const arma::vec& ub,
             const double rho,
             const int maxIter,
             const double relTol,
             const double absTol)
{
    //\rho=1;   //Arbitrary choice of the ADMM parameter rho.
    //x=z=u=zeros(n,1);   //Arbitrary starting points.
    //I=identity_matrix(n);
    //while (not converged)
    //    x=(ATA+ρI)−1(ATb+\rho(z-u));   //[1]
   // z=pos(x+u);
   // u=u+x-z;
   // end
    
    const int n = A.n_cols;
    const int sqrt_n = std::sqrt(n);
    
    arma::vec Atb= A.t()*b;
    arma::mat AtA = A.t()*A;
    
    AtA.diag() += rho;//update the diagonal
    
    arma::vec z_k = arma::zeros(n);
    arma::vec u_k = arma::zeros(n);
    arma::vec x_k = xopt;
    arma::vec z_old = z_k;
    arma::mat U = arma::chol(AtA);//AAt= U.t()*U
    
    double r_norm = std::numeric_limits<double>::max();
    double s_norm = r_norm;
    int it_k = 0;
    
    do{
        //prepare to solve by cholesky methods:
        arma::vec rhs = Atb + rho * (z_k - u_k);
        arma::vec tmp = arma::solve(arma::trimatl(arma::trans(U)), rhs);
        x_k = arma::solve(arma::trimatu(U), tmp);
        z_old = z_k;
        z_k = arma::min(arma::max(x_k + u_k, lb), ub);
        u_k = u_k + (x_k - z_k);
        r_norm = arma::norm(x_k - z_k, 2);
        s_norm = arma::norm(rho*(z_old - z_k), 2);
        it_k++;
        
        
    } while( (it_k < maxIter)
            ||
             (
              (r_norm > sqrt_n * absTol + relTol * std::max(arma::norm(x_k, 2), arma::norm(z_k, 2)))
             &&
             (s_norm  > sqrt_n * absTol + relTol * arma::norm(rho * u_k))
             )
            );
    xopt = z_k;
    return (it_k < maxIter)?0 : -1;
  }//qp
}//namespace arma::