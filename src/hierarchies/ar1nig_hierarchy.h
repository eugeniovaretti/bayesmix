#ifndef BAYESMIX_HIERARCHIES_AR1NIG_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_AR1NIG_HIERARCHY_H_

#include "base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "likelihoods/ar1_likelihood.h"
#include "priors/nig_prior_model.h"
#include "updaters/ar1nig_updater.h"
#include <math.h>

/**
 * Conjugate AR1-InverseGamma hierarchy for multivariate data.
 *
 * This class represents a hierarchical model where data are distributed
 * according to a AutoRegressive of order 1 likelihood (see the `AR1Likelihood` class for
 * details). The likelihood parameters have a Normal-InverseGamma centering
 * distribution (see the `NIGPriorModel` class for details). That is:
 *
 * \f[
 *    f(x_i \mid \bm{\mu}, \bm{\Sigma}) &= N_p(\mu,\sigma^2) \\
 *    (\mu,\sigma^2) & \sim NIG(\mu_0, \lambda_0, \alpha_0, \beta_0)
 * \f]
 *
 * The state is composed of mean and precision matrix that are built using univariate
 * mean and variance sampled from the prior/posterior. The state hyperparameters are
 * \f$(\mu_0, \lambda_0, \alpha_0, \beta_0)\f$, all scalar values. Note that
 * this hierarchy is conjugate, thus the marginal distribution is available in
 * closed form
 */
//CAMBIA TUTTO
class AR1NIGHierarchy
    : public BaseHierarchy<AR1NIGHierarchy, AR1Likelihood, NIGPriorModel> {
 public:
  AR1NIGHierarchy() = default;
  ~AR1NIGHierarchy() = default;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::AR1NIG;
  }

  //! Sets the default updater algorithm for this hierarchy
  void set_default_updater() { updater = std::make_shared<AR1NIGUpdater>(); }

  //! Initializes state parameters to appropriate values
  void initialize_state() override {
    // Initialize likelihood dimension to the time series length
    unsigned int ts_length = like.get_dataset()->cols();
    like->set_dim(ts_length);

    // Get hypers
    auto hypers = prior->get_hypers();
    // Initialize likelihood state
    State::UniLS state;
    state.mean = hypers.mean;
    state.var = hypers.scale / (hypers.shape + 1.0);
    like->set_state(state);
  };

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param hier_params  Pointer to the container of (prior or posterior)
  //! hyperparameter values
  //! @param datum        Point which is to be evaluated
  //! @return             The evaluation of the lpdf
  double marg_lpdf(ProtoHypersPtr hier_params,
                   const Eigen::RowVectorXd &datum) const override {
    auto params = hier_params->nnig_state();
    Eigen::MatrixXd data_sum_squares = datum.transpose() * datum;
    unsigned dim = datum.size();
    double a_p = params.var_scaling() + (data_sum_squares.diagonal(0).sum() - data_sum_squares.diagonal(0)[dim-1]) ;
    double b_p = params.var_scaling() * params.mean() + data_sum_squares.diagonal(1).sum();
    double c_p = params.var_scaling() * (params.mean() * params.mean()) + data_sum_squares.trace();
    double const_num = stan::math::tgamma(params.shape() + 0.5*dim);
    double det_sigma = pow(params.scale()/params.shape(), dim) * a_p/params.var_scaling();
    double const_den = stan::math::tgamma(params.shape()) *
              pow(stan::math::pi()*2*params.shape(), 0.5*dim) * pow(det_sigma,0.5);

    double factor3 = 1 + 1.0/(2.0*params.scale()) * (c_p - b_p*b_p/a_p);
    double c = const_num/const_den;

    double out = c * pow(factor3,-0.5*dim - params.shape());

    return std::log(out) ;
  };

};

#endif  // BAYESMIX_HIERARCHIES_AR1NIG_HIERARCHY_H_
