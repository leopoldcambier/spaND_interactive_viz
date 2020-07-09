#include <vector>
#include <cstdio>
#include <iostream>
#include <metis.h>
#include "mmio.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>

/**
 * g++ order.cpp -I /usr/local/Cellar/eigen/3.3.7/include/eigen3 -I ~/Softwares/metis/include -o order -std=c++14 ~/Softwares/metis/build/Darwin-x86_64/libmetis/libmetis.a -g -O0 -fsanitize=address -fsanitize=undefined
 */
int main(int argc, char** argv) {

    if(argc != 3) {
        printf("Need an input and output files\n");
        exit(1);
    }

    Eigen::SparseMatrix<double> A = mmio::sp_mmread<double,int>(argv[1]);
    int N = A.rows();
    std::vector<int> colptr;
    std::vector<int> rowval;
    colptr.push_back(0);
    for (int i = 0; i < N; i++) {
        int count = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,i); it; ++it) {
            if(it.row() != it.col()) {
                rowval.push_back(it.row());
                count++;
            }
        }
        colptr.push_back(rowval.size());
    }
    Eigen::VectorXi perm(N);
    Eigen::VectorXi iperm(N);
    int error = METIS_NodeND(&N,colptr.data(),rowval.data(),nullptr,nullptr,perm.data(),iperm.data());
    if(error != METIS_OK)
        printf("Error ? %d\n", error);
    mmio::dense_mmwrite<int>(argv[2], perm);
    printf("Done writing %s's permutation to %s\n", argv[1], argv[2]);
}

