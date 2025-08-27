#include "initializer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "checkerboard/checkerboard_base.h"
#include "checkerboard/cubic.h"
#include "checkerboard/square.h"
#include "lattice/cubic.h"
#include "lattice/honeycomb.h"
#include "lattice/lattice_base.h"
#include "lattice/square.h"
#include "measure/measure_handler.h"
#include "model/attractive_hubbard.h"
#include "model/model_base.h"
#include "model/repulsive_hubbard.h"
#include "svd_stack.h"
#include "utils/assert.h"
#include "walker.h"

namespace DQMC {

Context Initializer::parse_config(const Config& config) {
  ModelBasePtr model;
  LatticeBasePtr lattice;
  WalkerPtr walker;
  MeasureHandlerPtr meas_handler;
  CheckerBoardBasePtr checkerboard;
  // --------------------------------------------------------------------------------------------------
  //                                      Parse the Model module
  // --------------------------------------------------------------------------------------------------
  if (config.model_type == "RepulsiveHubbard") {
    model = std::make_unique<Model::RepulsiveHubbard>();
    model->set_model_params(config.hopping_t, config.onsite_u, config.chemical_potential);
  }

  // -----------------------------------  Attractive Hubbard model
  // -----------------------------------
  else if (config.model_type == "AttractiveHubbard") {
    model = std::make_unique<Model::AttractiveHubbard>();
    model->set_model_params(config.hopping_t, config.onsite_u, config.chemical_potential);
  }

  else {
    throw std::runtime_error(
        "DQMC::Initializer::parse_config(): "
        "undefined model type");
  }

  // --------------------------------------------------------------------------------------------------
  //                                    Parse the Lattice module
  // --------------------------------------------------------------------------------------------------
  if (config.lattice_type == "Square") {
    DQMC_ASSERT(config.lattice_size.size() == 2);

    // create 2d square lattice object
    lattice = std::make_unique<Lattice::Square>();
    lattice->set_lattice_params(config.lattice_size);
    lattice->initial();
  }

  else if (config.lattice_type == "Cubic") {
    DQMC_ASSERT(config.lattice_size.size() == 3);

    // create 3d cubic lattice object
    lattice = std::make_unique<Lattice::Cubic>();
    lattice->set_lattice_params(config.lattice_size);
    lattice->initial();
  }

  // ------------------------------------  2D Honeycomb lattice
  // --------------------------------------
  else if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  }

  else {
    throw std::runtime_error(
        "DQMC::Initializer::parse_config(): "
        "unsupported lattice type");
  }

  // --------------------------------------------------------------------------------------------------
  //                                  Parse the CheckerBoard module
  // --------------------------------------------------------------------------------------------------
  // note that the checkerboard method is currently only implemented for 2d
  // square lattice

  if (config.enable_checkerboard) {
    if (config.lattice_type == "Square") {
      checkerboard = std::make_unique<CheckerBoard::Square>();
    } else {
      throw std::runtime_error(
          "DQMC::Initializer::parse_config(): "
          "checkerboard is currently only implemented for "
          "2d square lattice");
    }
  }

  // --------------------------------------------------------------------------------------------------
  //                                   Parse the Walker module
  // --------------------------------------------------------------------------------------------------
  // create dqmc walker and set up parameters
  walker = std::make_unique<Walker>();
  walker->set_physical_params(config.beta, config.time_size);
  walker->set_stabilization_pace(config.stabilization_pace);

  // --------------------------------------------------------------------------------------------------
  //                                Parse the Measure Handler module
  // --------------------------------------------------------------------------------------------------

  // special observables, e.g. superfluid stiffness, are only supported for
  // specific lattice type.
  if (config.lattice_type != "Square") {
    if (std::find(config.observables.begin(), config.observables.end(), "superfluid_stiffness") !=
        config.observables.end()) {
      throw std::runtime_error("superfluid_stiffness is only supported for Square lattice");
    }
  }

  // create measure handler and set up parameters
  meas_handler = std::make_unique<Measure::MeasureHandler>();

  meas_handler->set_measure_params(config.sweeps_warmup, config.bin_num, config.bin_size,
                                   config.sweeps_between_bins);
  meas_handler->set_observables(config.observables);

  // --------------------------------------------------------------------------------------------------
  //                                Parse the input Momentum parmas
  // --------------------------------------------------------------------------------------------------
  // make sure that the lattice module is initialized ahead
  DQMC_ASSERT(lattice->InitialStatus());

  if (config.lattice_type == "Square") {
    // covert base class pointer to that of the derived square lattice class
    if (const auto square_lattice = dynamic_cast<const Lattice::Square*>(lattice.get())) {
      if (config.momentum == "GammaPoint") {
        meas_handler->set_measured_momentum(square_lattice->GammaPointIndex());
      } else if (config.momentum == "MPoint") {
        meas_handler->set_measured_momentum(square_lattice->MPointIndex());
      } else if (config.momentum == "XPoint") {
        meas_handler->set_measured_momentum(square_lattice->XPointIndex());
      } else {
        std::cerr << "DQMC::Initializer::parse_config(): " << "undefined momentum \'"
                  << config.momentum << "\' for 2d square lattice, " << "please check the config."
                  << std::endl;
        exit(1);
      }

      if (config.momentum_list == "KstarsAll") {
        meas_handler->set_measured_momentum_list(square_lattice->kStarsIndex());
      } else if (config.momentum_list == "DeltaLine") {
        meas_handler->set_measured_momentum_list(square_lattice->DeltaLineIndex());
      } else if (config.momentum_list == "ZLine") {
        meas_handler->set_measured_momentum_list(square_lattice->ZLineIndex());
      } else if (config.momentum_list == "SigmaLine") {
        meas_handler->set_measured_momentum_list(square_lattice->SigmaLineIndex());
      } else if (config.momentum_list == "Gamma2X2M2GammaLoop") {
        meas_handler->set_measured_momentum_list(square_lattice->Gamma2X2M2GammaLoopIndex());
      } else {
        std::cerr << "DQMC::Initializer::parse_config(): " << "undefined momentum list \'"
                  << config.momentum_list << "\' for 2d square lattice, "
                  << "please check the config." << std::endl;
        exit(1);
      }
    } else {
      std::cerr << "DQMC::Initializer::parse_config(): "
                << "fail to convert \'Lattice::LatticeBase\' to "
                   "\'Lattice::Square\'."
                << std::endl;
      exit(1);
    }
  }

  if (config.lattice_type == "Cubic") {
    // covert base class pointer to that of the derived cubic lattice class
    if (const auto cubic_lattice = dynamic_cast<const Lattice::Cubic*>(lattice.get())) {
      if (config.momentum == "GammaPoint") {
        meas_handler->set_measured_momentum(cubic_lattice->GammaPointIndex());
      } else if (config.momentum == "MPoint") {
        meas_handler->set_measured_momentum(cubic_lattice->MPointIndex());
      } else if (config.momentum == "XPoint") {
        meas_handler->set_measured_momentum(cubic_lattice->XPointIndex());
      } else if (config.momentum == "RPoint") {
        meas_handler->set_measured_momentum(cubic_lattice->RPointIndex());
      } else {
        std::cerr << "DQMC::Initializer::parse_config(): " << "undefined momentum \'"
                  << config.momentum << "\' for 3d cubic lattice, " << "please check the config."
                  << std::endl;
        exit(1);
      }

      if (config.momentum_list == "KstarsAll") {
        meas_handler->set_measured_momentum_list(cubic_lattice->kStarsIndex());
      } else if (config.momentum_list == "DeltaLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->DeltaLineIndex());
      } else if (config.momentum_list == "ZLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->ZLineIndex());
      } else if (config.momentum_list == "SigmaLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->SigmaLineIndex());
      } else if (config.momentum_list == "LambdaLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->LambdaLineIndex());
      } else if (config.momentum_list == "SLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->SLineIndex());
      } else if (config.momentum_list == "TLine") {
        meas_handler->set_measured_momentum_list(cubic_lattice->TLineIndex());
      } else {
        std::cerr << "DQMC::Initializer::parse_config(): " << "undefined momentum list \'"
                  << config.momentum_list << "\' for 3d cubic lattice, "
                  << "please check the config." << std::endl;
        exit(1);
      }
    } else {
      std::cerr << "DQMC::Initializer::parse_config(): "
                << "fail to convert \'Lattice::LatticeBase\' to \'Lattice::Cubic\'." << std::endl;
      exit(1);
    }
  }

  if (config.lattice_type == "Honeycomb") {
    throw std::runtime_error("Honeycomb lattice is not supported.");
  }

  return Context(std::move(model), std::move(lattice), std::move(walker), std::move(meas_handler),
                 std::move(checkerboard));
}

void Initializer::initial_modules(const Context& context) {
  // NOTE: the order of initializations below are important.
  DQMC_ASSERT(context.lattice->InitialStatus());

  context.handler->initial(*context.lattice, *context.walker);
  context.walker->initial(*context.lattice, *context.handler);
  context.model->initial(*context.lattice, *context.walker);

  if (context.checkerboard) {
    context.checkerboard->set_checkerboard_params(*context.lattice, *context.model,
                                                  *context.walker);
    context.checkerboard->initial();
    context.model->link(*context.checkerboard);
  } else {
    context.model->link();
  }
}

void Initializer::initial_dqmc(const Context& context) {
  // NOTE: this should be called after the initial configuration of the bosonic
  // fields.
  context.walker->initial_svd_stacks(*context.lattice, *context.model);
  context.walker->initial_greens_functions();
  context.walker->initial_config_sign();
}

}  // namespace DQMC
