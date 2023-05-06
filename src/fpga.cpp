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

int ComputeMG(const FPGAMatrix & A, const Vector & r, Vector & x) {
    CopyVector (r, x);
    return 0;
}

int ComputeSPMV( const FPGAMatrix & A, Vector & x, Vector & y) {
    assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
    assert(y.localLength>=A.localNumberOfRows);

    const double * const xv = x.values;
    double * const yv = y.values;
    const local_int_t nrow = A.localNumberOfRows;
    Row row;

    for (local_int_t i=0; i< nrow; i++)  {
        getRow(row, A, i);
        double sum = 0.0;
        const double * const cur_vals = row.values;
        const local_int_t * const cur_inds = row.indexes;
        const int cur_nnz = row.nonZeros;

        for (int j=0; j< cur_nnz; j++)
            sum += cur_vals[j]*xv[cur_inds[j]];
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

    assert(x.localLength>=n); // Test vector lengths
    assert(y.localLength>=n);

    const double * const xv = x.values;
    const double * const yv = y.values;
    double * const wv = w.values;

    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];

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
    assert(x.localLength>=n); // Test vector lengths
    assert(y.localLength>=n);

    double local_result = 0.0;
    double * xv = x.values;
    double * yv = y.values;

    if (yv==xv) {
        for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
    } else {
        for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
    }

    time_allreduce += 0.0;
    result = local_result;

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
int CG(const FPGAMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    bool doPreconditioning) {
    
    doPreconditioning = false; // for now, it must be disabled

    //double t_begin = mytimer();  // Start timing right away
    normr = 0.0;
    double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

    local_int_t nrow = A.localNumberOfRows;
    Vector & r = data.r; // Residual vector
    Vector & z = data.z; // Preconditioned residual vector
    Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
    Vector & Ap = data.Ap;

    //if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
    int print_freq = 1;
    if (print_freq>50) print_freq=50;
    if (print_freq<1)  print_freq=1;
#endif
    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);  TOCK(t2); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif

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
            TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
        } else {
            oldrtz = rtz;
            TICK(); ComputeDotProduct (nrow, r, z, rtz, t4, A.isDotProductOptimized); TOCK(t1); // rtz = r'*z
            beta = rtz/oldrtz;
            TICK(); ComputeWAXPBY (nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);  TOCK(t2); // p = beta*p + z
        }

        TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
        TICK(); ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized); TOCK(t1); // alpha = p'*Ap
        alpha = rtz/pAp;
        TICK(); ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized);// x = x + alpha*p
                ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);  TOCK(t2);// r = r - alpha*Ap
        TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
        normr = sqrt(normr);
        #ifdef HPCG_DEBUG
        if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
            HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
        #endif
        niters = k;
    }

    // Store times
    //times[1] += t1; // dot-product time
    //times[2] += t2; // WAXPBY time
    //times[3] += t3; // SPMV time
    //times[4] += t4; // AllReduce time
    //times[5] += t5; // preconditioner apply time
    //#ifndef HPCG_NO_MPI
    //  times[6] += t6; // exchange halo time
    //#endif
   // times[0] += mytimer() - t_begin;  // Total time. All done...
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
                 double* bValues, const int bLength, double* xValues, const int xLength,
                 const int maxIters, double* testNormsValues, const int numberOfCgSets) {

#pragma HLS INTERFACE m_axi port=AValues offset=slave bundle=aximm1
#pragma HLS INTERFACE m_axi port=AIndexes offset=slave bundle=aximm2
#pragma HLS INTERFACE m_axi port=ANonZeros offset=slave bundle=aximm3
#pragma HLS INTERFACE m_axi port=bValues offset=slave bundle=aximm4
#pragma HLS INTERFACE m_axi port=xValues offset=slave bundle=aximm5
#pragma HLS INTERFACE m_axi port=testNormsValues offset=slave bundle=aximm6

#pragma HLS INTERFACE s_axilite port=AValues
#pragma HLS INTERFACE s_axilite port=AIndexes
#pragma HLS INTERFACE s_axilite port=ANonZeros
#pragma HLS INTERFACE s_axilite port=NumOfRows
#pragma HLS INTERFACE s_axilite port=NumOfColumns
#pragma HLS INTERFACE s_axilite port=bValues
#pragma HLS INTERFACE s_axilite port=bLength
#pragma HLS INTERFACE s_axilite port=xValues
#pragma HLS INTERFACE s_axilite port=xLength
#pragma HLS INTERFACE s_axilite port=maxIters
#pragma HLS INTERFACE s_axilite port=testNormsValues
#pragma HLS INTERFACE s_axilite port=numberOfCgSets
#pragma HLS INTERFACE s_axilite port=return

        FPGAMatrix A = constructMatrix(AValues, AIndexes, ANonZeros, NumOfRows, NumOfColumns);
        Vector b;
        b.values = bValues;
        b.localLength = bLength;
        Vector x;
        x.values = xValues;
        x.localLength = xLength;
        double optTolerance = 0.0;  // Force maxIters iterations
        double normr = 0.0;
        double normr0 = 0.0;
        int niters = 0;
        int ierr;
        CGData data;
        double r[NUM_OF_ROWS];
        double z[NUM_OF_COLS];
        double p[NUM_OF_COLS];
        double Ap[NUM_OF_ROWS];
        data.r.values = r;
        data.z.values = z;
        data.p.values = p;
        data.Ap.values = Ap;
        data.r.localLength = NumOfRows;
        data.z.localLength = NumOfColumns;
        data.p.localLength = NumOfColumns;
        data.Ap.localLength = NumOfRows;

        for (int i = 0; i < numberOfCgSets; ++i) {
            ZeroVector(x); // Zero out x
            ierr = CG( A, data, b, x, maxIters, optTolerance, niters, normr, normr0, true);
            //if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
            //if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
            testNormsValues[i] = normr/normr0; // Record scaled residual from this run
        }
    }
}
