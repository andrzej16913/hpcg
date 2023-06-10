//
// Created by andrzejradzik on 10.04.23.
//

// STL includes
#include <cmath>
#include <cassert>

// HPCG includes
#include "Vector.hpp"
#include "CGData.hpp"
//#include "ComputeSPMV.hpp"
//#include "ComputeMG.hpp"
//#include "ComputeDotProduct.hpp"
//#include "ComputeWAXPBY.hpp"

// HPCG-FPGA inlcudes
#include "fpga.hpp"

using namespace fpga;

void mytimer(void) {}

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  mytimer() //!< record current time in 't0'
#define TOCK(t) mytimer() //!< store time difference in 't' using time in 't0'

double SYMGS_step(const FPGAMatrix& A, double* const xv, double rhs, int i) {
    Row row;
    getRow(row, A, i);
    const double * const currentValues = row.values;
    const local_int_t * const currentColIndices = row.indexes;
    const int currentNumberOfNonzeros = row.nonZeros;
    double currentDiagonal = 1.0; // Dummy diagonal value
    double sum = rhs; // RHS value
    double tmp[27];
    double dummy = 0.0;

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        dummy = currentValues[j];
        if (curCol == i) {
            currentDiagonal = dummy;
            dummy = 0.0;
        } else {
            dummy *= xv[curCol];
        }
        tmp[j] = dummy;
    }

    for (int j = 0; j < currentNumberOfNonzeros; j++ ) {
        sum -= tmp[j];
    }

    return sum / currentDiagonal;
}

/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.


  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
int ComputeSYMGS(const FPGAMatrix & A, const Vector & r, Vector & x) {
    const local_int_t nrow = A.localNumberOfRows;
    const double * const rv = r.values;
    double * const xv = x.values;

    for (int i = 0; i < nrow; i++) {
        xv[i] = SYMGS_step(A, xv, rv[i], i);
    }

    // Now the back sweep.

    for (int i = nrow - 1; i >= 0; i--) {
        xv[i] = SYMGS_step(A, xv, rv[i], i);
    }

    return 0;
}

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
int ComputeMG(const FPGAMatrix & A, const Vector & r, Vector & x) {
     ZeroVector(x); // initialize x to zero

    int ierr = 0;
    ierr = ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;

    return 0;
}

int ComputeSPMV( const FPGAMatrix & A, Vector & x, Vector & y) {
    const double * const xv = x.values;
    double * const yv = y.values;
    const local_int_t nrow = A.localNumberOfRows;
    Row row;
    double tmp[27];

    for (local_int_t i=0; i< nrow; i++)  {
        getRow(row, A, i);
        double sum = 0.0;
        const double * const cur_vals = row.values;
        const local_int_t * const cur_inds = row.indexes;
        const int cur_nnz = row.nonZeros;

        for (int j=0; j < cur_nnz; j++) {
            tmp[j] = cur_vals[j] * xv[cur_inds[j]];
        }

        for (int j=0; j< cur_nnz; j++)
            sum += tmp[j];

        yv[i] = sum;
    }
    return 0;
}

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
                  const double beta, const Vector & y, Vector & w, bool optimized) {

    const double * const xv = x.values;
    const double * const yv = y.values;
    double * const wv = w.values;

    for (local_int_t i=0; i<n; i++)
        wv[i] = alpha * xv[i] + beta * yv[i];

    return 0;
}

/*!
  Routine to compute the dot product of two vectors where:

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
                          double & result, double & time_allreduce, bool optimized) {

    double * xv = x.values;
    double * yv = y.values;
/*    double tmp[8];

    for (int j = 0; j < 8; j++) {
        tmp[j] = 0.0;
    }

    for (local_int_t i=0; i<n; i += 8) {
        for (int j = 0; j < 8; ++j) {
            tmp[j] += xv[i+j] * yv[i+j];
        }
    }

    tmp[0] += tmp[1];
    tmp[2] += tmp[3];
    tmp[4] += tmp[5];
    tmp[6] += tmp[7];

    tmp[0] += tmp[2];
    tmp[4] += tmp[6];

    time_allreduce += 0.0;
    result = tmp[0] + tmp[4];
*/
    double tmp0 = 0.0;
    double tmp1 = 0.0;
    double tmp2 = 0.0;
    double tmp3 = 0.0;
    double tmp4 = 0.0;
    double tmp5 = 0.0;
    double tmp6 = 0.0;
    double tmp7 = 0.0;

    for (local_int_t i=0; i<n; i += 8) {
        tmp0 += xv[i] * yv[i];
        tmp1 += xv[i + 1] * yv[i + 1];
        tmp2 += xv[i + 2] * yv[i + 2];
        tmp3 += xv[i + 3] * yv[i + 3];
        tmp4 += xv[i + 4] * yv[i + 4];
        tmp5 += xv[i + 5] * yv[i + 5];
        tmp6 += xv[i + 6] * yv[i + 6];
        tmp7 += xv[i + 7] * yv[i + 7];
    }

    tmp0 += tmp1;
    tmp2 += tmp3;
    tmp4 += tmp5;
    tmp6 += tmp7;

    tmp0 += tmp2;
    tmp4 += tmp6;

    result = tmp0 + tmp4;

    return 0;
}

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const FPGAMatrix & A, CGData & data, const Vector & b, Vector & x, Vector& r2, Vector& p2,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    bool doPreconditioning) {

    normr = 0.0;
    double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

    local_int_t nrow = A.localNumberOfRows;
    Vector & r = data.r; // Residual vector
    Vector & z = data.z; // Preconditioned residual vector
    Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
    Vector & Ap = data.Ap;

    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);  TOCK(t2); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
    normr = sqrt(normr);

    // Record initial residual for convergence testing
    normr0 = normr;

    // Start iterations
    // Convergence check accepts an error of no more than 6 significant digits of tolerance
    for (int k=1; k<=max_iter && normr/normr0 > tolerance * (1.0 + 1.0e-6); k++ ) {
        TICK();
        if (doPreconditioning)
            ComputeMG(A, r, z); // Apply preconditioner
        else
            CopyVector (r, z); // copy r to z (no preconditioning)
        TOCK(t5); // Preconditioner apply time

        if (k == 1) {
            TICK(); ComputeWAXPBY(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized); TOCK(t2); // Copy Mr to p
            TICK(); ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
        } else {
            oldrtz = rtz;
            TICK(); ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
            beta = rtz/oldrtz;
            ComputeWAXPBY(nrow, 1.0, z, beta, p, p2, A.isWaxpbyOptimized);  TOCK(t2); // p = beta*p + z
            CopyVector(p2, p); // p2 is used as buffer, it speeds up computation a bit
        }

        TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
        TICK(); ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized); TOCK(t1); // alpha = p'*Ap
        alpha = rtz/pAp;
        TICK(); ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized);// x = x + alpha*p
                ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r2, A.isWaxpbyOptimized);  TOCK(t2);// r = r - alpha*Ap
                CopyVector(r2, r); // same as above
        TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
        normr = sqrt(normr);
        #ifdef HPCG_DEBUG
        if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
            HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
        #endif
        niters = k;
    }


    return 0;
}

/*!
  Main kernel
  Translates data to HPCG internal format and runs CG procedure in a loop

  @param[inout] AValues    The known system matrix values
  @param[inout] AIndexes   The known system matrix indexes
  @param[inout] ANonZeros  The known system matrix non-zeros in a row
  @param[in]    NumOfRows  The known system matrix number of rows
  @param[in]    NumOfColumns  The known system matrix number of columns
  @param[in]    bValues    The known right hand side vector values
  @param[in]    bLength    The known right hand side vector length
  @param[inout] xValues    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    xLength    Length of x vector
  @param[in]    maxIters  The maximum number of iterations to perform, even if tolerance is not met.
  @param[out]   testNormsValues The array of rescaled residuals
  @param[in]    numberOfCgStes  The number of runs in the loop

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/

extern "C" {
    void run_CG (double* AValues, int* AIndexes, char* ANonZeros, const int NumOfRows, const int NumOfColumns,
                 double* bValues, double* xValues, double* rValues,
                 double* zValues, double* pValues, double* ApValues,
                 double* r2Values, double* p2Values, bool doPreconditioning,
                 const int maxIters, double* testNormsValues, const int numberOfCgSets) {

#pragma HLS INTERFACE m_axi port=AValues offset=slave bundle=aximm1
#pragma HLS INTERFACE m_axi port=AIndexes offset=slave bundle=aximm2
#pragma HLS INTERFACE m_axi port=ANonZeros offset=slave bundle=aximm3
#pragma HLS INTERFACE m_axi port=bValues offset=slave bundle=aximm4
#pragma HLS INTERFACE m_axi port=xValues offset=slave bundle=aximm5
#pragma HLS INTERFACE m_axi port=rValues offset=slave bundle=aximm6
#pragma HLS INTERFACE m_axi port=zValues offset=slave bundle=aximm7
#pragma HLS INTERFACE m_axi port=pValues offset=slave bundle=aximm8
#pragma HLS INTERFACE m_axi port=ApValues offset=slave bundle=aximm9
#pragma HLS INTERFACE m_axi port=r2Values offset=slave bundle=aximm10
#pragma HLS INTERFACE m_axi port=p2Values offset=slave bundle=aximm11
#pragma HLS INTERFACE m_axi port=testNormsValues offset=slave bundle=aximm12

#pragma HLS INTERFACE s_axilite port=AValues
#pragma HLS INTERFACE s_axilite port=AIndexes
#pragma HLS INTERFACE s_axilite port=ANonZeros
#pragma HLS INTERFACE s_axilite port=NumOfRows
#pragma HLS INTERFACE s_axilite port=NumOfColumns
#pragma HLS INTERFACE s_axilite port=bValues
#pragma HLS INTERFACE s_axilite port=xValues
#pragma HLS INTERFACE s_axilite port=rValues
#pragma HLS INTERFACE s_axilite port=zValues
#pragma HLS INTERFACE s_axilite port=pValues
#pragma HLS INTERFACE s_axilite port=ApValues
#pragma HLS INTERFACE s_axilite port=r2Values
#pragma HLS INTERFACE s_axilite port=p2Values
#pragma HLS INTERFACE x_axilite port=doPreconditioning
#pragma HLS INTERFACE s_axilite port=maxIters
#pragma HLS INTERFACE s_axilite port=testNormsValues
#pragma HLS INTERFACE s_axilite port=numberOfCgSets
#pragma HLS INTERFACE s_axilite port=return

        FPGAMatrix A = constructMatrix(AValues, AIndexes, ANonZeros, NumOfRows, NumOfColumns);
        Vector b;
        b.values = bValues;
        b.localLength = NumOfRows;
        Vector x;
        x.values = xValues;
        x.localLength = NumOfRows;
        double optTolerance = 0.0;  // Force maxIters iterations
        double normr = 0.0;
        double normr0 = 0.0;
        int niters = 0;
        int ierr;
        CGData data;
        data.r.values = rValues;
        data.z.values = zValues;
        data.p.values = pValues;
        data.Ap.values = ApValues;
        data.r.localLength = NumOfRows;
        data.z.localLength = NumOfColumns;
        data.p.localLength = NumOfColumns;
        data.Ap.localLength = NumOfRows;

        Vector r2;
        r2.values = r2Values;
        r2.localLength = NumOfRows;
        Vector p2;
        p2.values = p2Values;
        p2.localLength = NumOfColumns;

        for (int i = 0; i < numberOfCgSets; ++i) {
            ZeroVector(x); // Zero out x
            ierr = CG( A, data, b, x, r2, p2, maxIters, optTolerance, niters, normr, normr0, doPreconditioning);
            testNormsValues[i] = normr/normr0; // Record scaled residual from this run
        }
    }
}
