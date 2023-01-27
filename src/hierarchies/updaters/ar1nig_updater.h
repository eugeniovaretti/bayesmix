#ifndef BAYESMIX_HIERARCHIES_UPDATERS_AR1NIG_UPDATER_H_
#define BAYESMIX_HIERARCHIES_UPDATERS_AR1NIG_UPDATER_H_

#include "semi_conjugate_updater.h"
#include "src/hierarchies/likelihoods/ar1_likelihood.h"
#include "src/hierarchies/priors/nig_prior_model.h"

/**
 * Updater specific for the `AR1Likelihood` used in combination
 * with `NIGPriorModel`, that is the model
 *
 * \f[
 *      \boldsymbol{y_i} \mid \mu, \sigma^2 &\stackrel{\small\mathrm{iid}}{\sim} N(\mu,
 * \sigma^2) \\
 *      \mu \mid \sigma^2 &\sim N(\mu_0, \sigma^2 / \lambda) \\
 *      \sigma^2 &\sim InvGamma(a, b)
 * \f]
 *
 * It exploits the conjugacy of the model to sample the full conditional of
 * \f$ (\mu, \sigma^2) \f$ by calling `NIGPriorModel::sample` with updated
 * parameters
 */

class AR1NIGUpdater
    : public SemiConjugateUpdater<AR1Likelihood, NIGPriorModel> {
 public:
  AR1NIGUpdater() = default;
  ~AR1NIGUpdater() = default;



  bool is_conjugate() const override { return true; };

  ProtoHypersPtr compute_posterior_hypers(AbstractLikelihood &like,
                                          AbstractPriorModel &prior) override;

  std::shared_ptr<AbstractUpdater> clone() const override {
    auto out =
        std::make_shared<AR1NIGUpdater>(static_cast<AR1NIGUpdater const &>(*this));
    out->clear_hypers();
    return out;
  }
};

#endif  // BAYESMIX_HIERARCHIES_UPDATERS_AR1NIG_UPDATER_H_
