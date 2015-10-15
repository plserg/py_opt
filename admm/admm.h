//
//  admm.h
//  armadillo_test
//
//  Created by Sergey Plyasunov on 10/5/15.
//  Copyright (c) 2015 Sergey Plyasunov. All rights reserved.
//

#ifndef armadillo_test_admm_h
#define armadillo_test_admm_h

#include <armadillo>
namespace admm{
//
// solves argmin{||y-Hx||^2_2}
//s.t. lb<=x<=ub
//
int qp(arma::vec& xopt,/*solution out*/
             const arma::mat& H,
             const arma::vec& y,
             const arma::vec& lb,
             const arma::vec& ub,
             const double rho = 1.0,
             const int maxIter=100,
             const double relTol=1e-3,
             const double absTol=1e-3
             );
}
#endif
