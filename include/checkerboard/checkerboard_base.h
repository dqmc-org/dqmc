#pragma once

/**
 *  This header file defines the base class CheckerBoard::Base,
 *  for implementing the checkerboard breakups on biparticle lattice.
 *  It is pure virtual, and the breakups for any specific lattice should be
 * derived from this class.
 */

#include <Eigen/Core>

namespace Model {
class ModelBase;
}
namespace Lattice {
class LatticeBase;
}
namespace DQMC {
class Walker;
}

namespace CheckerBoard {

using Site = std::vector<int>;
using Matrix = Eigen::MatrixXd;
using RealScalar = double;

using ModelBase = Model::ModelBase;
using LatticeBase = Lattice::LatticeBase;
using Walker = DQMC::Walker;

// ------------------------- Pure virtual base class
// CheckerBoard::CheckerBoardBase -----------------------------
class CheckerBoardBase {
 public:
  CheckerBoardBase() = default;
  virtual ~CheckerBoardBase() = default;

  // initialize from lattice, model and dqmcWalker
  virtual void set_checkerboard_params(const LatticeBase &lattice,
                                       const ModelBase &model,
                                       const Walker &walker) = 0;
  virtual void initial() = 0;

  // multiply the exponent of hopping matrix K using checkerboard breakups
  virtual void mult_expK_from_left(Matrix &matrix) const = 0;
  virtual void mult_expK_from_right(Matrix &matrix) const = 0;
  virtual void mult_inv_expK_from_left(Matrix &matrix) const = 0;
  virtual void mult_inv_expK_from_right(Matrix &matrix) const = 0;
  virtual void mult_trans_expK_from_left(Matrix &matrix) const = 0;
};
}  // namespace CheckerBoard
