#include "dirichlet_c2_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"
#include "src/utils/rng.h"

void DirichletC2Mixing::update_state(
    const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
    const std::vector<unsigned int> &allocations) {
  auto &rng = bayesmix::Rng::Instance().get();
  auto priorcast = cast_prior();
  unsigned int n = allocations.size();

  if (priorcast->has_fixed_value()) {
    return;
  }
/*
  else if (priorcast->has_gamma_prior()) {
    // Recover parameters
    unsigned int k = unique_values.size();
    double alpha = priorcast->gamma_prior().totalmass_prior().shape();
    double beta = priorcast->gamma_prior().totalmass_prior().rate();
    // Update state (see Neal (2000) for details)
    double phi = stan::math::gamma_rng(state.totalmass + 1, n, rng);
    double odds = (alpha + k - 1) / (n * (beta - log(phi)));
    double prob = odds / (1 + odds);
    double p = stan::math::uniform_rng(0.0, 1.0, rng);
    if (p <= prob) {
      state.totalmass = stan::math::gamma_rng(alpha + k, beta - log(phi), rng);
    } else {
      state.totalmass =
          stan::math::gamma_rng(alpha + k - 1, beta - log(phi), rng);
    }
  }
  */

  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}



double DirichletC2Mixing::mass_existing_cluster(
    const unsigned int n, const unsigned int n_clust, const bool log,
    const bool propto, const std::shared_ptr<AbstractHierarchy> hier,
    const Eigen::RowVectorXd &covariate) const {
  double out;
  // suppongo che un parametro in proto sia "a" (distanza hard boundary)
  // implemento hard boundary
  bool is_near = 1;
  unsigned n_data_clus = covariates_ptr->rows();
  for(size_t i = 0; i < n_data_clus; ++i)
  {
    // funzione che calcola distanza tra due RowVectorXd (forse il primo
    // è da converitre a vettore)
    if((covariates_ptr->row(i)-covariate).norm() > state.a)
    {
      //std::cout << "The unit is far from datum " << i << std::endl;
      is_near = 0;
      break;
    }
  }

  if (log) {
    out = hier->get_log_card();
    if (!propto) out -= std::log(n + state.totalmass);
    return is_near ? out : std::log(0);
  } else {
    out = 1.0 * hier->get_card();
    if (!propto) out /= (n + state.totalmass);
    return is_near ? out : 0;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double DirichletC2Mixing::mass_new_cluster(const unsigned int n,
                                         const unsigned int n_clust,
                                         const bool log,
                                         const bool propto,
                                         const Eigen::RowVectorXd &covariate) const {
  double out;
  if (log) {
    out = state.logtotmass;
    if (!propto) out -= std::log(n + state.totalmass);
  } else {
    out = state.totalmass;
    if (!propto) out /= (n + state.totalmass);
  }
  return out;
}



void DirichletC2Mixing::set_state_from_proto(
    const google::protobuf::Message &state_) {
  auto &statecast = downcast_state(state_);
  state.totalmass = statecast.dpc2_state().totalmass();
  state.logtotmass = std::log(state.totalmass);
  state.a = statecast.dpc2_state().a();
}


std::shared_ptr<bayesmix::MixingState> DirichletC2Mixing::get_state_proto()
    const {
  bayesmix::DPC2State state_;
  state_.set_totalmass(state.totalmass);
  state_.set_a(state.a);
  auto out = std::make_shared<bayesmix::MixingState>();
  out->mutable_dpc2_state()->CopyFrom(state_);
  return out;
}

void DirichletC2Mixing::initialize_state() {
  auto priorcast = cast_prior();
  if (priorcast->has_fixed_value()) {
    state.totalmass = priorcast->fixed_value().totalmass();
    state.logtotmass = std::log(state.totalmass);
    state.a = priorcast->fixed_value().a();
    if (state.totalmass <= 0) {
      throw std::invalid_argument("Total mass (or a) parameter must be > 0");
    }
    if (state.a <= 0) {
      throw std::invalid_argument("Distance parameter a must be > 0");
    }
  }
/*
  else if (priorcast->has_gamma_prior()) {
    double alpha = priorcast->gamma_prior().totalmass_prior().shape();
    double beta = priorcast->gamma_prior().totalmass_prior().rate();
    if (a <= 0) {
      throw std::invalid_argument("Distance parameter a must be > 0");
    }
    if (alpha <= 0) {
      throw std::invalid_argument("Shape parameter must be > 0");
    }
    if (beta <= 0) {
      throw std::invalid_argument("Rate parameter must be > 0");
    }
    state.totalmass = alpha / beta;
  }
*/
  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}
