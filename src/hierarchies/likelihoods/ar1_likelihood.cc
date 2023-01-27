#include "ar1_likelihood.h"

#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"
/*
void set_state(const State &state_, bool update_card = true) {
  state = state_;
  if (update_card) {
    set_card(state.card);
  }
};
*/
/*
double AR1Likelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum) const {
  // Decido di non tenere salvate le matrici ma di crearle ogni volta,
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);

  //se si vuole mettere questo in una funzione sample_var
  //std::cout << "mean: " <<  state.mean << " --- state.var =  " << state.var << std::endl;
  Eigen::VectorXd diag = (1.0+state.mean*state.mean)/state.var * Eigen::VectorXd::Ones(dim);
  diag[dim-1] = 1.0/state.var;
  //std::cout << "diag: \n" << diag << std::endl;
  Eigen::VectorXd codiag = -state.mean/state.var * Eigen::VectorXd::Ones(dim-1);
  //std::cout << "codiag: \n" << codiag << std::endl;
  Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(dim,dim);
  Omega.diagonal(0) = diag;
  Omega.diagonal(1) = codiag;
  Omega.diagonal(-1) = codiag;

  //std::cout << "Omega: \n" << Omega << std::endl;

  //Eigen::LLT<Eigen::MatrixXd> lltOfA(Omega);
  //Eigen::MatrixXd prec_chol = lltOfA.matrixL();
  Eigen::MatrixXd prec_chol = Eigen::LLT<Eigen::MatrixXd>(Omega).matrixL();
  //std::cout << "prec_chol: \n" << prec_chol << std::endl;
  Eigen::VectorXd diagL = prec_chol.diagonal();
  double prec_logdet = 2 * log(diagL.array()).sum();

  //std::cout << "prec_logdet: \n" << prec_logdet << std::endl;
  //std::cout << "codiag: \n" << codiag << std::endl;

  return bayesmix::multi_normal_prec_lpdf(datum, mean, prec_chol,
                                          prec_logdet);
}
*/
void AR1Likelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           bool add) {
  // Check if dim is not defined yet (this usually doesn't happen if the
  // hierarchy is initialized)
  if (!dim) set_dim(datum.size());
  // Updates
  if (add) {
    //data_sum += datum.transpose();
    data_sum_squares += datum.transpose() * datum;
  } else {
    //data_sum -= datum.transpose();
    data_sum_squares -= datum.transpose() * datum;
  }
}

void AR1Likelihood::clear_summary_statistics() {
  //data_sum = Eigen::VectorXd::Zero(dim);
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}

void AR1Likelihood::set_state(const State::UniLS &state_, bool update_card) {
  state = state_;
  if (update_card) {
    set_card(state.card);
  }

  mean = Eigen::VectorXd::Zero(dim);

  //se si vuole mettere questo in una funzione sample_var
  //std::cout << "mean: " <<  state.mean << " --- state.var =  " << state.var << std::endl;
  Eigen::VectorXd diag = (1.0+state.mean*state.mean)/state.var * Eigen::VectorXd::Ones(dim);
  diag[dim-1] = 1.0/state.var;
  //std::cout << "diag: \n" << diag << std::endl;
  Eigen::VectorXd codiag = -state.mean/state.var * Eigen::VectorXd::Ones(dim-1);
  //std::cout << "codiag: \n" << codiag << std::endl;
  Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(dim,dim);
  Omega.diagonal(0) = diag;
  Omega.diagonal(1) = codiag;
  Omega.diagonal(-1) = codiag;

  //std::cout << "Omega: \n" << Omega << std::endl;

  //Eigen::LLT<Eigen::MatrixXd> lltOfA(Omega);
  //Eigen::MatrixXd prec_chol = lltOfA.matrixL();
  prec_chol = Eigen::LLT<Eigen::MatrixXd>(Omega).matrixL();
  //std::cout << "prec_chol: \n" << prec_chol << std::endl;
  Eigen::VectorXd diagL = prec_chol.diagonal();
  prec_logdet = 2 * log(diagL.array()).sum();
}

double AR1Likelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum) const {
  return bayesmix::multi_normal_prec_lpdf(datum, mean, prec_chol,
                                          prec_logdet);
}
