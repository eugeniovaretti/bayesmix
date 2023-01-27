#include "ar1nig_updater.h"

#include "src/hierarchies/likelihoods/states/includes.h"
#include "src/hierarchies/priors/hyperparams.h"

AbstractUpdater::ProtoHypersPtr AR1NIGUpdater::compute_posterior_hypers(
    AbstractLikelihood& like, AbstractPriorModel& prior) {
  // Likelihood and Prior downcast
  auto& likecast = downcast_likelihood(like);
  auto& priorcast = downcast_prior(prior);

  // Getting required quantities from likelihood and prior
  int card = likecast.get_card();
  int dim = likecast.get_dim();

  Eigen::MatrixXd data_sum_squares = likecast.get_data_sum_squares();
  auto hypers = priorcast.get_hypers();

  // No update possible
  if (card == 0) {
    return priorcast.get_hypers_proto();
  }

  // Compute posterior hyperparameters
  double mean, var_scaling, shape, scale;
  //Definisco quantità A,B,C dei Conti...
  // per ottimizzazione usa queste due quantita' sotto
  //const auto& diag_ss = data_sum_squares.diagonal(0)
  // const auto& codiag_ss = data_sum_squares.diagonal(1)
  double a_p = hypers.var_scaling + (data_sum_squares.diagonal(0).sum() - data_sum_squares.diagonal(0)[dim-1]) ;
  double b_p = hypers.var_scaling * hypers.mean + data_sum_squares.diagonal(1).sum();
  double c_p = hypers.var_scaling * (hypers.mean * hypers.mean) + data_sum_squares.trace();

  mean = b_p/a_p;
  var_scaling = a_p;
  shape = hypers.shape + 0.5 * card * dim ;
  scale = hypers.scale - 0.5 * (b_p*b_p/a_p - c_p);

  // Proto conversion
  ProtoHypers out;
  out.mutable_nnig_state()->set_mean(mean);
  out.mutable_nnig_state()->set_var_scaling(var_scaling);
  out.mutable_nnig_state()->set_shape(shape);
  out.mutable_nnig_state()->set_scale(scale);
  return std::make_shared<ProtoHypers>(out);
}
