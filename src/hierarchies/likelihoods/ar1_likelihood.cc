#include "ar1_likelihood.h"

#include "src/utils/distributions.h"
#include "src/utils/eigen_utils.h"
#include "src/utils/proto_utils.h"

void AR1Likelihood::update_sum_stats(const Eigen::RowVectorXd &datum,
                                           bool add) {
  // Check if dim is not defined yet (this usually doesn't happen if the
  // hierarchy is initialized)
  if (!dim) set_dim(datum.size());
  // Updates
  if (add) {
    data_sum_squares += datum.transpose() * datum;
  } else {
    data_sum_squares -= datum.transpose() * datum;
  }
}

void AR1Likelihood::clear_summary_statistics() {
  data_sum_squares = Eigen::MatrixXd::Zero(dim, dim);
}

void AR1Likelihood::set_state(const State::UniLS &state_, bool update_card) {
  state = state_;
  if (update_card) {
    set_card(state.card);
  }

  mean = Eigen::VectorXd::Zero(dim);


  Eigen::VectorXd diag = (1.0+state.mean*state.mean)/state.var * Eigen::VectorXd::Ones(dim);
  diag[dim-1] = 1.0/state.var;
  Eigen::VectorXd codiag = -state.mean/state.var * Eigen::VectorXd::Ones(dim-1);
  Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(dim,dim);
  Omega.diagonal(0) = diag;
  Omega.diagonal(1) = codiag;
  Omega.diagonal(-1) = codiag;

  prec_chol = Eigen::LLT<Eigen::MatrixXd>(Omega).matrixL();
  Eigen::VectorXd diagL = prec_chol.diagonal();
  prec_logdet = 2 * log(diagL.array()).sum();
}

double AR1Likelihood::compute_lpdf(
    const Eigen::RowVectorXd &datum) const {
  return bayesmix::multi_normal_prec_lpdf(datum, mean, prec_chol,
                                          prec_logdet);
}
