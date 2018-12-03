#include <iostream>
#include <cmath>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

using namespace std;
using namespace dlib;

typedef matrix<double, 0, 1> column_vector;

double loss(const column_vector& beta,
            const matrix<double>& X,
            const matrix<double>& y,
            int N) {

    double val = -(1/(double)N)*(trans(trans(X)*y)*beta);

    column_vector vec = X*beta;
    for (int i=0; i<N; i++) {
        val += (1/(double)N)*log(1 + exp(vec(i)));
    }

    return val;
}

column_vector prob(const column_vector& beta,
            const matrix<double>& X,
            const matrix<double>& y,
            int N) {
    
    matrix<double> probs(N,1);
    double prod;

    for (int i=0; i<N; i++) {
        prod = rowm(X,i)*beta;
        if (std::isnan(exp(-1*prod))) {
            probs(i) = 1;
        }
        else {
            probs(i) = 1/(1+exp(-1*prod));
        }
    }

    return probs;
}

column_vector d_loss(const column_vector& beta,
                     const matrix<double>& X,
                     const matrix<double>& y,
                     int N) {

    matrix<double> p(N,1);
    p = prob(beta, X, y, N);

    return (1/(double)N)*trans(X)*(p - y);
}

double beta_iter_obj_func(const column_vector& beta,
                          const column_vector& z,
                          const column_vector& u,
                          const matrix<double>& X,
                          const matrix<double>& y,
                          int N,
                          double rho) {

    column_vector temp = beta - z + u;
    
    return loss(beta, X, y, N) + (rho/2)*(trans(temp)*temp);
}

/* functor for use as objective function in beta iterations */
struct beta_iter_obj_func_parametrized {
  public:
    beta_iter_obj_func_parametrized(const column_vector& z,
                                    const column_vector& u,
                                    const matrix<double>& X,
                                    const matrix<double>& y,
                                    int N,
                                    double rho) : 
        z(z), u(u), X(X), y(y), N(N), rho(rho) {}

    double operator()(const column_vector& beta) const {
        return beta_iter_obj_func(beta, z, u, X, y, N, rho);
    }

  private:
    column_vector z, u;
    matrix<double> X, y;
    int N;
    double rho;
};

column_vector beta_iter_obj_func_gradient(const column_vector& beta,
                                          const column_vector& z,
                                          const column_vector& u,
                                          const matrix<double>& X,
                                          const matrix<double>& y,
                                          int N,
                                          double rho) {

    return d_loss(beta, X, y, N) + rho*(beta - z + u);
}

/* functor for use as gradient in beta iterations */
struct beta_iter_obj_func_gradient_parametrized {
  public:
    beta_iter_obj_func_gradient_parametrized(const column_vector& z,
                                             const column_vector& u,
                                             const matrix<double>& X,
                                             const matrix<double>& y,
                                             int N,
                                             double rho) :
        z(z), u(u), X(X), y(y), N(N), rho(rho) {}

    column_vector operator()(const column_vector& beta) const {
        return beta_iter_obj_func_gradient(beta, z, u, X, y, N, rho);
    }

  private:
    column_vector z, u;
    matrix<double> X, y;
    int N;
    double rho;
};

/* z update functions */
double soft_threshold(double x, double c) {
    if (x > c) {
        return x-c;
    }
    else if (abs(x) <= c) {
        return 0;
    }
    else {
        return x+c;
    }
}

column_vector z_iter(const column_vector& beta_av,
                     const column_vector& u_av,
                     int p,                             /* # of predictors */
                     double lambda,
                     double rho,
                     int num_pes) {
    column_vector z_new;
    z_new = zeros_matrix<double>(p+1,1);
    for (int i=0; i<p+1; i++) {
        if (i == 0) {
            z_new(i) = beta_av(i) + u_av(i);
        }
        else {
            z_new(i) = soft_threshold(beta_av(i) + u_av(i),
                                      lambda/(rho*num_pes));
        }
    }

    return z_new;
}

/* u update */
column_vector u_iter(const column_vector& beta_curr,    /* current beta */
                     const column_vector& z_curr,       /* current z */
                     const column_vector& u_prev) {     /* previous u */
    return u_prev + beta_curr - z_curr;
}













/* end */
