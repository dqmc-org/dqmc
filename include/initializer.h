#pragma once

/**
 *  This header file defines the dqmc initializer class
 * DQMC::Initializer, which contains static member functions to
 * initialize the dqmc modules altogether.
 */

#include <memory>
#include <string_view>
#include <vector>

namespace Lattice {
class LatticeBase;
}
namespace Model {
class ModelBase;
}
namespace Measure {
class MeasureHandler;
}
namespace CheckerBoard {
class CheckerBoardBase;
}

namespace DQMC {

// forward declaration
class Walker;

using LatticeBase = Lattice::LatticeBase;
using ModelBase = Model::ModelBase;
using CheckerBoardBase = CheckerBoard::CheckerBoardBase;
using MeasureHandler = Measure::MeasureHandler;

using LatticeBasePtr = std::unique_ptr<Lattice::LatticeBase>;
using ModelBasePtr = std::unique_ptr<Model::ModelBase>;
using CheckerBoardBasePtr = std::unique_ptr<CheckerBoard::CheckerBoardBase>;
using MeasureHandlerPtr = std::unique_ptr<Measure::MeasureHandler>;
using WalkerPtr = std::unique_ptr<Walker>;

using MomentumIndex = int;
using MomentumIndexList = std::vector<int>;

// ----------------------- Interface class DQMC::Initializer
// ------------------------
class Initializer {
 public:
  // parse parmameters from the toml configuration file
  // create modules and setup module parameters according to the input
  // configurations
  static void parse_toml_config(std::string_view toml_config, int world_size,
                                ModelBasePtr& model, LatticeBasePtr& lattice,
                                WalkerPtr& walker,
                                MeasureHandlerPtr& meas_handler,
                                CheckerBoardBasePtr& checkerboard);

  // initialize modules including Lattice, Model, Walker and MeasureHandler
  // without checkerboard breakups.
  static void initial_modules(ModelBase& model, LatticeBase& lattice,
                              Walker& walker, MeasureHandler& meas_handler);

  // initialize modules including Lattice, Model, Walker and MeasureHandler
  // with checkerboard breakups.
  static void initial_modules(ModelBase& model, LatticeBase& lattice,
                              Walker& walker, MeasureHandler& meas_handler,
                              CheckerBoardBase& checkerboard);

  // prepare for the dqmc simulation,
  // especially initializing the greens functions and SVD stacks
  static void initial_dqmc(ModelBase& model, LatticeBase& lattice,
                           Walker& walker, MeasureHandler& meas_handler);
};
}  // namespace DQMC
