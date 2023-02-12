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
  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}



double DirichletC2Mixing::mass_existing_cluster(
    const unsigned int n, const unsigned int n_clust, const bool log,
    const bool propto, const std::shared_ptr<AbstractHierarchy> hier,
    const Eigen::RowVectorXd &covariate) const {
  double out;
  // flag on the closeness
  bool is_near = 1;
  std::set<int> index_clusters = hier->get_data_idx();
  for(const auto i : index_clusters)
  {
    if(haversine_formula(covariates_ptr->row(i),covariate) > state.a)
    {
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
  else {
    throw std::invalid_argument("Unrecognized mixing prior");
  }
}

// This function compute the haversine formula (distance in km) between two points
// given as a vector (lat,lon)
double haversine_formula(const Eigen::RowVectorXd& point1, const Eigen::RowVectorXd& point2)
{
  // check if the coordinate is bidimensional
  if(point1.cols() != 2 || point2.cols() != 2)
  {
    std::cerr << "Coordinates are not in the right format!" << std::endl;
  }
  const double earth_radius = 6371.0;
  double lat1 = point1(0) * M_PI / 180.0;
  double lat2 = point2(0) * M_PI / 180.0;
  double delta_lat = (lat2 - lat1);
  double delta_long = (point2(1) - point1(1)) * M_PI / 180.0;

  double a = std::sin(delta_lat / 2.0) * std::sin(delta_lat / 2.0) +
           std::cos(lat1) * std::cos(lat2) *
           std::sin(delta_long / 2.0) * std::sin(delta_long / 2.0);

  double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));

  return earth_radius * c;
}
