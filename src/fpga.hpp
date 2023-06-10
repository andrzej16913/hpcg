// By Andrzej Radzik in 2023

// fpga.hpp
// This file contains constants requiered for compilation on xrt devices

#ifndef HPCG_FPGA_HPP
#define HPCG_FPGA_HPP

constexpr int NX = 104;
constexpr int NY = 104;
constexpr int NZ = 104;

// Maximum number of rows in matrix A
constexpr long long NUM_OF_ROWS = NX * NY * NZ;

// Maximum number of columns in matrix A
constexpr long long NUM_OF_COLS = NUM_OF_ROWS;

// Maximum number of non-zero values in a row in a matrix
constexpr int MAX_ROW_LENGTH = 27;

// Maximum size of Matrix
constexpr long long MATRIX_SIZE = NUM_OF_ROWS * MAX_ROW_LENGTH;

constexpr int MAX_RUN_COUNT = 1024 * 1024;

namespace fpga {
    /*
     * Replacement for SparseMatrix
     */
    struct FPGAMatrix {
        double* values;
        int* indexes;
        char* nonZeros;
        int localNumberOfRows;
        int localNumberOfColumns;
        bool isWaxpbyOptimized;
        bool isDotProductOptimized;
    };

    struct Row {
        double* values;
        int* indexes;
        char nonZeros;
    };

    inline FPGAMatrix constructMatrix(double* values, int* indexes, char* nonZeros,
                               int localNumberOfRows, int localNumberOfColumns) {
        FPGAMatrix matrix;
        matrix.values = values;
        matrix.indexes = indexes;
        matrix.nonZeros = nonZeros;
        matrix.localNumberOfRows = localNumberOfRows;
        matrix.localNumberOfColumns = localNumberOfColumns;
        matrix.isWaxpbyOptimized = false;
        matrix.isDotProductOptimized = false;
        return matrix;
    }

    inline void getRow(Row& row, FPGAMatrix matrix, int index) {
        int i = index * MAX_ROW_LENGTH;
        row.values = matrix.values + i;
        row.indexes = matrix.indexes + i;
        row.nonZeros = matrix.nonZeros[index];
    }
}

#endif // HPCG_FPGA_HPP

