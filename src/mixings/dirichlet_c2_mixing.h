#ifndef BAYESMIX_MIXINGS_DIRICHLET_C2_MIXING_H_
#define BAYESMIX_MIXINGS_DIRICHLET_C2_MIXING_H_

#include <google/protobuf/message.h>

#include <memory>
#include <stan/math/rev.hpp>
#include <vector>

#include "base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "src/hierarchies/abstract_hierarchy.h"

namespace DirichletC2 {
struct State {
  double totalmass, logtotmass, a;
};
};  // namespace Dirichlet with C2 cohesion function

/**
 * Class that represents the EPPF induced by the Dirithclet process (DP)
 * introduced in Ferguson (1973), see also Sethuraman (1994) and a Cohesion function,
 * C2, introduced in Quintana-Page (2016), that is:
 * SCRIVI FORMULA
 * The EPPF induced by the DP depends on a `totalmass` parameter M, and a
 * distance parameter "a".
 * Given a clustering of n elements into k clusters, each with cardinality
 * \f$ n_j, j=1, ..., k \f$ the EPPF of the DP gives the following
 * probabilities for the cluster membership of the (n+1)-th observation:
 *
 * \f[ **SCRIVI FORMULA
 *    p(\text{j-th cluster} | ...) &= n_j / (n + M) \\
 *    p(\text{new cluster} | ...) &= M / (n + M)
 * \f]
 *
 * The state is composed of M and a, but we also store log(M) for efficiency
 * reasons. For more information about the class, please refer instead to base
 * classes, `AbstractMixing` and `BaseMixing`.
 */

class DirichletC2Mixing
    : public BaseMixing<DirichletC2Mixing, DirichletC2::State, bayesmix::DPC2Prior> {
 public:
  DirichletC2Mixing() = default;
  ~DirichletC2Mixing() = default;

  //! Performs conditional update of state, given allocations and unique values
  //! @param unique_values  A vector of (pointers to) Hierarchy objects
  //! @param allocations    A vector of allocations label
  void update_state(
      const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
      const std::vector<unsigned int> &allocations) override;

  //! Read and set state values from a given Protobuf message
  void set_state_from_proto(const google::protobuf::Message &state_) override;

  //! Writes current state to a Protobuf message and return a shared_ptr
  //! New hierarchies have to first modify the field 'oneof val' in the
  //! MixingState message by adding the appropriate type
  std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::MixingId get_id() const override { return bayesmix::MixingId::DPC2; }

  //! Returns whether the mixing is conditional or marginal
  bool is_conditional() const override { return false; }
  bool is_dependent() const override {return true; }

 protected:
  //! Returns probability mass for an old cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param n_clust    Number of clusters
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param hier       `Hierarchy` object representing the cluster
  //! @return           Probability value
  double mass_existing_cluster(
      const unsigned int n, const unsigned int n_clust, const bool log,
      const bool propto,
      const std::shared_ptr<AbstractHierarchy> hier,
      const Eigen::RowVectorXd &covariate) const override;

  //! Returns probability mass for a new cluster (for marginal mixings only)
  //! @param n          Total dataset size
  //! @param log        Whether to return logarithm-scale values or not
  //! @param propto     Whether to include normalizing constants or not
  //! @param n_clust    Current number of clusters
  //! @return           Probability value
  double mass_new_cluster(const unsigned int n, const unsigned int n_clust,
                          const bool log, const bool propto,
                          const Eigen::RowVectorXd &covariate) const override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;
};

#endif  // BAYESMIX_MIXINGS_DIRICHLET_C2_MIXING_H_
