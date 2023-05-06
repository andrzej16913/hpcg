//
// Created by andrzejradzik on 10.04.23.
//

// STL includes
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

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
               std::vector<double> times, bool doPreconditioning) {

    // Computing sizes of buffers
    constexpr size_t AValuesSize = sizeof(double) * MATRIX_SIZE;
    constexpr size_t AIndexesSize = sizeof(int) * MATRIX_SIZE;
    constexpr size_t ANonZerosSize = sizeof(char) * NUM_OF_ROWS;
    constexpr size_t vectorSize = sizeof(double) * NUM_OF_ROWS;
    constexpr size_t normsSize = sizeof(double) * MAX_RUN_COUNT;

    // Definitions of settings
    std::ostream& outStream = std::cerr;
    std::string binaryFile = "./run_CG.xclbin";
    int device_index = 0;
    int error = 0;

    outStream << AValuesSize << std::endl;
    outStream << AIndexesSize << std::endl;
    outStream << "Number of rows: " << A.localNumberOfRows << ", " << b.localLength << std::endl;
    outStream << type_name<decltype( A.mtxIndL[0][0])>() << std::endl;
    outStream << "mgData: " << A.mgData << std::endl;
    outStream << "maxIter: " << maxIter << std::endl;
    outStream << "Ac: "<< std::endl;
    auto* old = &A;
    for (auto ptr = 0; ptr < 4; ptr++) {
        outStream << old->localNumberOfRows << std::endl;
        old = old->Ac;
    }

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
    //if (error > 0) return error;

    // Prepare the device and the kernel
    outStream << "Open the device: " << device_index << std::endl;
    auto device = xrt::device(device_index);
    outStream << "Load the xclbin: " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto kernel = xrt::kernel(device, uuid, "run_CG", xrt::kernel::cu_access_mode::exclusive);

    // Create buffers for data transfer
    outStream << "Allocate Buffer Values in Global Memory\n";
    auto boAValues = xrt::bo(device, AValuesSize, 6);
    outStream << "Allocate Buffer Indexes in Global Memory\n";
    auto boAIndexes = xrt::bo(device, AIndexesSize, 1);
    outStream << "Allocate Buffer NonZeros in Global Memory\n";
    auto boANonZeros = xrt::bo(device, ANonZerosSize, 2);
    outStream << "Allocate Buffer vector b in Global Memory\n";
    auto bobVector = xrt::bo(device, vectorSize,3);
    outStream << "Allocate Buffer vector x in Global Memory\n";
    auto boxVector = xrt::bo(device, vectorSize, 4);
    outStream << "Allocate Buffer norms in Global Memory\n";
    auto boNorms = xrt::bo(device, normsSize, 5);

    outStream << type_name<decltype(boAValues)>() << std::endl;

    // Map the contents of the buffer object into host memory
    auto boAValuesMap = boAValues.map<double*>();
    auto boAIndexesMap = boAIndexes.map<int*>();
    auto boANonZerosMap = boANonZeros.map<char*>();
    auto bobVectorMap = bobVector.map<double*>();
    auto boxVectorMap = boxVector.map<double*>();
    auto boNormsMap = boNorms.map<double*>();

    outStream << type_name<decltype(boAValuesMap)>() << std::endl;
    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;

    // Copy matrix A and vector b to buffer
    double* valuesBegin = new double[MATRIX_SIZE];
    int* indexesBegin = new int[MATRIX_SIZE];
    double* valuesPointer = valuesBegin;
    int* indexesPointer = indexesBegin;
    for (size_t i = 0; i < A.localNumberOfRows; ++i) {
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

    outStream << "b[0]: " << b.values[0] << ", " << bobVectorMap[0] << std::endl;
    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;

    // Synchronize buffer content with device side
    outStream << "Synchronize input buffer data to device global memory\n";
    boAValues.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    outStream << "Synchronize input buffer data to device global memory\n";
    boAIndexes.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    outStream << "Synchronize input buffer data to device global memory\n";
    boANonZeros.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    outStream << "Synchronize input buffer data to device global memory\n";
    bobVector.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;

    // Execute kernel and measure total time
    outStream << "Execution of the kernel\n";
    double startTime = mytimer();
    auto run = kernel(boAValues, boAIndexes, boANonZeros, A.localNumberOfRows, A.localNumberOfColumns,
                      bobVector, b.localLength, boxVector, x.localLength,
                      maxIter, boNorms, testNormsData.samples);
    run.wait();
    double stopTime = mytimer();

    outStream << "Runs: " << testNormsData.samples << std::endl;
    outStream << "Time: " << stopTime - startTime << std::endl;
    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;
    outStream << "x[0]: " << x.values[0] << ", " << boxVectorMap[0] << std::endl;

    // Synchronize the output from the device
    outStream << "Get the output data from the device" << std::endl;
    boAValues.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    outStream << "Get the output data from the device" << std::endl;
    boAIndexes.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    outStream << "Get the output data from the device" << std::endl;
    boANonZeros.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    outStream << "Get the output data from the device" << std::endl;
    boxVector.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    outStream << "Get the output data from the device" << std::endl;
    boNorms.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;
    outStream << "x[0]: " << x.values[0] << ", " << boxVectorMap[0] << std::endl;

    // Copy the output to HPCG data structures
    valuesPointer = valuesBegin;
    indexesPointer = indexesBegin;
    std::copy(boAValuesMap, boAValuesMap + A.localNumberOfRows * MAX_ROW_LENGTH, valuesBegin);
    std::copy(boAIndexesMap, boAIndexesMap + A.localNumberOfRows * MAX_ROW_LENGTH, indexesBegin);
    for (size_t i = 0; i < A.localNumberOfRows; ++i) {
        std::copy(valuesPointer, valuesPointer + MAX_ROW_LENGTH, A.matrixValues[i]);
        std::copy(indexesPointer, indexesPointer + MAX_ROW_LENGTH, A.mtxIndL[i]);
        valuesPointer += MAX_ROW_LENGTH;
        indexesPointer += MAX_ROW_LENGTH;
    }

    std::copy(boANonZerosMap, boANonZerosMap + A.localNumberOfRows, A.nonzerosInRow);
    std::copy(boxVectorMap, boxVectorMap + x.localLength, x.values);
    std::copy(boNormsMap, boNormsMap + testNormsData.samples, testNormsData.values);

    outStream << "Test norm: " << boNormsMap[0] << ", " << testNormsData.values[0] << std::endl;
    outStream << "x[0]: " << x.values[0] << ", " << boxVectorMap[0] << std::endl;

    // Print computed norms to HPCG_fout
    for (size_t i = 0; i < testNormsData.samples; ++i) {
        outStream << "Call [" << i << "] Scaled Residual [" << testNormsData.values[i] << "]" << std::endl;
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

    return 0;
}
