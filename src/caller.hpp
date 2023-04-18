//
// Created by andrzejradzik on 10.04.23.
//

#ifndef HPCG_CALLER_HPP
#define HPCG_CALLER_HPP

#include <vector>
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "TestNorms.hpp"

/*!
    Procedure, which translates HPCG data structures into XRT compatible data,
    calls kernel which executes CG function in a loop, and measures execution time.

    @param[inout] A         The known system matrix
    @param[in]    b         The known right hand side vector
    @param[inout] x         On entry: the initial guess; on exit: the new approximate solution
    @param[in]    maxIter   The maximum number of iterations to perform, even if tolerance is not met.
    @param[inout] testNormsData   The data of test norms
    @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
    @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

    @return Returns zero on success and a non-zero value otherwise
*/

int callKernel(const SparseMatrix& A, const Vector& b, Vector& x,
                const int maxIter, TestNormsData testNormsData,
                std::vector<double> times, bool doPreconditioning);

#endif //HPCG_CALLER_HPP
