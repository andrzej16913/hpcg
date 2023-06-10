//
// Created by andrzejradzik on 10.04.23.
//

// STL includes
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// XRT includes
#include <experimental/xrt_xclbin.h>
//#include "xrt/xrt_bo.h"
//#include "xrt/xrt_device.h"
//#include "xrt/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// HPCG includes
#include "hpcg.hpp"
#include "mytimer.hpp"
#include "SparseMatrix.hpp"
#include "TestNorms.hpp"
#include "Vector.hpp"

// HPCG-FPGA specific includes
#include "caller.hpp"
#include "fpga.hpp"

int callKernel(const SparseMatrix & A, const Vector & b, Vector & x,
               const int maxIter, TestNormsData testNormsData,
               std::vector<double>& times, bool doPreconditioning) {

    // Computing sizes of buffers
    constexpr size_t AValuesSize = sizeof(double) * MATRIX_SIZE;
    constexpr size_t AIndexesSize = sizeof(int) * MATRIX_SIZE;
    constexpr size_t ANonZerosSize = sizeof(char) * NUM_OF_ROWS;
    constexpr size_t vectorSize = sizeof(double) * NUM_OF_ROWS;
    constexpr size_t normsSize = sizeof(double) * MAX_RUN_COUNT;
    constexpr size_t horizontalSize = sizeof(double) * NUM_OF_COLS;

    // Definitions of settings, you can change them if you need
    std::ostream& outStream = std::cerr;
    std::string binaryFile = "./run_CG.xclbin";
    int device_index = 1;
    int error = 0;

    // Check if input data is not too large
    if (A.localNumberOfRows > NUM_OF_ROWS) {
        outStream << "ERROR: Number of rows in matrix A is too large: " << A.localNumberOfRows << std::endl;
        error = 1;
    }
    if (b.localLength > NUM_OF_ROWS) {
        outStream << "ERROR: Length of vector b is too large: " << b.localLength << std::endl;
        error = 2;
    }
    if (x.localLength > NUM_OF_ROWS) {
        outStream << "ERROR: Length of vector x is too large: " << x.localLength << std::endl;
        error = 3;
    }
    if (testNormsData.samples > MAX_RUN_COUNT) {
        outStream << "ERROR: Number of samples is too large: " << testNormsData.samples << std::endl;
        error = 4;
    }
    if (error > 0) return error;

    // Prepare the device and the kernel
    auto device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binaryFile);
    auto kernel = xrt::kernel(device, uuid, "run_CG", xrt::kernel::cu_access_mode::exclusive);

    // Create buffers for data transfer
    auto boAValues = xrt::bo(device, AValuesSize, 6);
    auto boAIndexes = xrt::bo(device, AIndexesSize, 1);
    auto boANonZeros = xrt::bo(device, ANonZerosSize, 2);
    auto bobVector = xrt::bo(device, vectorSize, 3);
    auto boxVector = xrt::bo(device, vectorSize, 4);
    auto boNorms = xrt::bo(device, normsSize, 5);
    auto borVector = xrt::bo(device, vectorSize, 7);
    auto bozVector = xrt::bo(device, horizontalSize, 8);
    auto bopVector = xrt::bo(device, horizontalSize, 9);
    auto boApVector = xrt::bo(device, vectorSize, 10);
    auto bor2Vector = xrt::bo(device, vectorSize, 11);
    auto bop2Vector = xrt::bo(device, horizontalSize, 12);

    // Map the contents of the buffer object into host memory
    auto boAValuesMap = boAValues.map<double*>();
    auto boAIndexesMap = boAIndexes.map<int*>();
    auto boANonZerosMap = boANonZeros.map<char*>();
    auto bobVectorMap = bobVector.map<double*>();
    auto boxVectorMap = boxVector.map<double*>();
    auto boNormsMap = boNorms.map<double*>();

    // Copy matrix A and vector b to buffer
    double* valuesBegin = new double[MATRIX_SIZE];
    int* indexesBegin = new int[MATRIX_SIZE];
    double* valuesPointer = valuesBegin;
    int* indexesPointer = indexesBegin;
    for (int i = 0; i < A.localNumberOfRows; ++i) {
        std::copy(A.matrixValues[i], A.matrixValues[i] + MAX_ROW_LENGTH, valuesPointer);
        std::copy(A.mtxIndL[i], A.mtxIndL[i] + MAX_ROW_LENGTH, indexesPointer);
        valuesPointer += MAX_ROW_LENGTH;
        indexesPointer += MAX_ROW_LENGTH;
    }

    std::fill(boAValuesMap, boAValuesMap + MATRIX_SIZE, 0);
    std::fill(boAIndexesMap, boAIndexesMap + MATRIX_SIZE, 0);
    std::copy(valuesBegin, valuesPointer, boAValuesMap);
    std::copy(indexesBegin, indexesPointer, boAIndexesMap);
    std::copy(A.nonzerosInRow, A.nonzerosInRow + A.localNumberOfRows, boANonZerosMap);
    std::copy(b.values, b.values + b.localLength, bobVectorMap);

    // Synchronize buffer content with device side
    boAValues.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boAIndexes.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boANonZeros.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bobVector.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute kernel and measure total time
    outStream << "Execution of the kernel\n";
    double startTime = mytimer();
    auto run = kernel(boAValues, boAIndexes, boANonZeros, A.localNumberOfRows, A.localNumberOfColumns,
                      bobVector, boxVector, borVector, bozVector, bopVector, boApVector,
                      bor2Vector, bop2Vector, doPreconditioning,
                      maxIter, boNorms, testNormsData.samples);
    run.wait();
    double stopTime = mytimer();

    // Synchronize the output from the device
    boAValues.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boAIndexes.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boANonZeros.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boxVector.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boNorms.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Copy the output to HPCG data structures
    valuesPointer = valuesBegin;
    indexesPointer = indexesBegin;
    std::copy(boAValuesMap, boAValuesMap + A.localNumberOfRows * MAX_ROW_LENGTH, valuesBegin);
    std::copy(boAIndexesMap, boAIndexesMap + A.localNumberOfRows * MAX_ROW_LENGTH, indexesBegin);
    for (int i = 0; i < A.localNumberOfRows; ++i) {
        std::copy(valuesPointer, valuesPointer + MAX_ROW_LENGTH, A.matrixValues[i]);
        std::copy(indexesPointer, indexesPointer + MAX_ROW_LENGTH, A.mtxIndL[i]);
        valuesPointer += MAX_ROW_LENGTH;
        indexesPointer += MAX_ROW_LENGTH;
    }

    std::copy(boANonZerosMap, boANonZerosMap + A.localNumberOfRows, A.nonzerosInRow);
    std::copy(boxVectorMap, boxVectorMap + x.localLength, x.values);
    std::copy(boNormsMap, boNormsMap + testNormsData.samples, testNormsData.values);

    // Print computed norms to HPCG_fout
    for (int i = 0; i < testNormsData.samples; ++i) {
        HPCG_fout << "Call [" << i << "] Scaled Residual [" << testNormsData.values[i] << "]" << std::endl;
    }

    // Write total execution and dummy times to times array
    for (size_t i = 1; i < 7; ++i) {
        times[i] = 0.0;
    }

    times[0] = stopTime - startTime;
    outStream << times[0] << std::endl;

    delete[] valuesBegin;
    delete[] indexesBegin;

    return error;
}
