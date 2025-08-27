#pragma once

/**
 *  This header file defines the dqmc initializer class
 * DQMC::Initializer, which contains static member functions to
 * initialize the dqmc modules altogether.
 */

#include <memory>
#include <utility>
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

struct Context {
  ModelBasePtr model;
  LatticeBasePtr lattice;
  WalkerPtr walker;
  MeasureHandlerPtr handler;
  CheckerBoardBasePtr checkerboard;

  Context(ModelBasePtr m, LatticeBasePtr l, WalkerPtr w, MeasureHandlerPtr h,
          CheckerBoardBasePtr cb)
      : model(std::move(m)),
        lattice(std::move(l)),
        walker(std::move(w)),
        handler(std::move(h)),
        checkerboard(std::move(cb)) {}

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(Context&&) = default;
};

// ----------------------- Interface class DQMC::Initializer
// ------------------------
class Initializer {
 public:
  struct Config {
    // Model
    std::string model_type;
    double hopping_t;
    double onsite_u;
    double chemical_potential;

    // Lattice
    std::string lattice_type;
    std::vector<int> lattice_size;

    // Checkerboard
    bool enable_checkerboard;

    // Monte carlo
    double beta;
    double time_size;
    int stabilization_pace;

    // Measure
    int sweeps_warmup;
    int bin_num;
    int bin_size;
    int sweeps_between_bins;

    // Observables
    std::vector<std::string> observables;

    // Momentum parameters
    std::string momentum;
    std::string momentum_list;
  };

  // Parses parameters from the config struct, creates all modules,
  // and returns them in an owning Context object.
  static Context parse_config(const Config& config);

  // Initializes modules using the provided context.
  static void initial_modules(const Context& context);

  // Prepares for the DQMC simulation (e.g., initializing Green's functions).
  static void initial_dqmc(const Context& context);
};
}  // namespace DQMC
