#ifndef BAYESMIX_HIERARCHIES_LIKELIHOODS_AR1_LIKELIHOOD_H_
#define BAYESMIX_HIERARCHIES_LIKELIHOODS_AR1_LIKELIHOOD_H_

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <vector>

#include "algorithm_state.pb.h"
#include "base_likelihood.h"
#include "states/includes.h"

/**
 * An autoregressive of order1 likelihood, using the `State::UniLS` state.
 * Represents the model:
 *
 * \f[
 *    \bm{y}_1,\dots, \bm{y}_k \stackrel{\small\mathrm{iid}}{\sim}
 * N_p(\bm{\mu}, \Sigma), \f]
 *
 * where \f$ (\bm{\mu}, \Sigma) \f$ are stored in a `State::UniLS` state.
 * The sufficient statistic stored is the sum of the \f$ \bm{y}_i^T \bm{y}_i \f$.
 */

class AR1Likelihood
    : public BaseLikelihood<AR1Likelihood, State::UniLS> {
 public:
  AR1Likelihood() = default;
  ~AR1Likelihood() = default;
  bool is_multivariate() const override { return true; };
  bool is_dependent() const override { return false; };
  void clear_summary_statistics() override;

  // dim reflects the length of the time series
  void set_dim(unsigned int dim_) {
    dim = dim_;
    clear_summary_statistics();
  };

  void set_state(const State::UniLS &state_, bool update_card = true);

  unsigned int get_dim() const { return dim; };

  Eigen::MatrixXd get_data_sum_squares() const { return data_sum_squares; };

 protected:
  double compute_lpdf(const Eigen::RowVectorXd &datum) const override;
  void update_sum_stats(const Eigen::RowVectorXd &datum, bool add) override;

  unsigned int dim;
  //this matrix is useful to calculate A,B,C
  Eigen::MatrixXd data_sum_squares;

  Eigen::VectorXd mean;
  Eigen::MatrixXd prec_chol;
  double prec_logdet;

};

#endif  // BAYESMIX_HIERARCHIES_LIKELIHOODS_AR1_LIKELIHOOD_H_
