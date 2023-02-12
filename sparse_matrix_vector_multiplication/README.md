# Sparse matrix-vector multiplication
Task: Write a program for sparse matrix vector multiplication, try to optimize the performance
with tiling and unrolling ideas discussed in context of dense matrices. 

Input data:
Sparse matrix: copy the file from https://sparse.tamu.edu/SNAP/higgs-twitter. The first row of
the input file is the number of rows, number of columns, number of nonzeros of the sparse
matrix Each of the following row represents a nonzero as row_index, column_index, value.
Dense vector: assume all elements in the dense vector are 1.

Requirements:
1) Store the sparse matrix in CSR or COO, see
https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_
Yale_format) for more details.
2) Write the program in C or C++, compile the program with g++
3) Use ‘gettimeofday’ to measure the execution time of the loop.
4) Try to divide the sparse matrix into column blocks, and see if the performance is
improved.