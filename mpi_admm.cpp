#include <iostream>
#include <string>
#include <mpi.h>
#include "admm.h"

using namespace std;

/* utility functions for use with MPI_Reduce and MPI_Bcast */
void vector_to_array(const column_vector&, double*, int);
void array_to_vector(column_vector&, const double*, int);

int
main(int argc, char *argv[])
{
    int N = atoi(argv[1]);              /* dimension of EACH data chunk */
    int p = atoi(argv[2]);
    int iters = atoi(argv[3]);
    double rho = atof(argv[4]);
    double lambda = atof(argv[5]);

    int rank, num_pes;
    double local_beta[p+1], global_beta[p+1];
    double global_z[p+1], local_u[p+1], global_u[p+1];

/* 
 * Begin parallel program
 */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* non-root ranks */
    if (rank != 0) {
        /* read rank-specific data chunk into X and y */
        string filename = "data_";
        filename += to_string(rank);
        filename += ".txt";
        ifstream file(filename);

        matrix<double> X(N,p+1);
        matrix<double> y(N,1);
        for (int i=0; i<N; i++) {
            for (int j=0; j<p+1; j++) {
                file >> X(i,j);
            }
            file >> y(i);
        }
        file.close();

        /* carry out iterations */
        column_vector beta, z, u;
        beta = z = u = zeros_matrix<double>(p+1,1);

        for (int i=0; i<iters; i++) {
            cout << "Rank " << rank << ": iteration " << i+1 << endl;

            /* beta update */
            beta_iter_obj_func_parametrized f(z, u, X, y, N, rho);
            beta_iter_obj_func_gradient_parametrized df(z, u, X, y, N, rho);

            find_min(lbfgs_search_strategy(10),
                     objective_delta_stop_strategy(1e-7),
                     f, df, beta, -1);

            /* z update -- send beta to rank 0, then receive z from rank 0 */
            vector_to_array(beta, local_beta, p+1);
            MPI_Reduce(&local_beta, &global_beta, p+1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
            MPI_Bcast(&global_z, p+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            array_to_vector(z, global_z, p+1);

            /* u update -- update u using z, send u to rank 0 */
            u = u_iter(beta, z, u);
            vector_to_array(u, local_u, p+1);
            MPI_Reduce(&local_u, &global_u, p+1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
        }
    }

    /* rank 0 */
    else {
            column_vector beta_av, u_av, z;
            beta_av = u_av = z = zeros_matrix<double>(p+1,1);

            /* NOTE: we MUST set local_beta and local_u to 0 to avoid
             * undefined behavior in MPI_Reduce */
            vector_to_array(beta_av, local_beta, p+1);
            vector_to_array(u_av, local_u, p+1);

       for (int i=0; i<iters; i++) {
            /* gather beta, compute z, and broadcast z update */
            MPI_Reduce(&local_beta, &global_beta, p+1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
            array_to_vector(beta_av, global_beta, p+1);
            beta_av = (1/(double)(num_pes-1))*beta_av;
            z = z_iter(beta_av, u_av, p, lambda, rho, num_pes);
            vector_to_array(z, global_z, p+1);
            MPI_Bcast(&global_z, p+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            /* gather u and calculate u_av */
            MPI_Reduce(&local_u, &global_u, p+1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);
            array_to_vector(u_av, global_u, p+1);
            u_av = (1/(double)(num_pes-1))*u_av;
        }

        cout << "Beta:\n" << beta_av << endl;
    }

    MPI_Finalize();

    return 0;
}

void vector_to_array(const column_vector& v, double* arr, int length) {
    for (int i=0; i<length; i++) {
        arr[i] = v(i);
    }
}

void array_to_vector(column_vector& v, const double* arr, int length) {
    for (int i=0; i<length; i++) {
        v(i) = arr[i];
    }
}
