#include "initializer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

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

void Initializer::parse_config(const Config& config, int world_size, ModelBasePtr& model,
                               LatticeBasePtr& lattice, WalkerPtr& walker,
                               MeasureHandlerPtr& meas_handler, CheckerBoardBasePtr& checkerboard) {
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

    // initial lattice module in place
    if (!lattice->InitialStatus()) {
      lattice->initial();
    }
  }

  else if (config.lattice_type == "Cubic") {
    DQMC_ASSERT(config.lattice_size.size() == 3);

    // create 3d cubic lattice object
    lattice = std::make_unique<Lattice::Cubic>();
    lattice->set_lattice_params(config.lattice_size);

    // initial lattice module in place
    if (!lattice->InitialStatus()) {
      lattice->initial();
    }
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

  // send measuring tasks to a set of processes
  const int bins_per_proc = (config.bin_num % world_size == 0) ? config.bin_num / world_size
                                                               : config.bin_num / world_size + 1;
  meas_handler->set_measure_params(config.sweeps_warmup, bins_per_proc, config.bin_size,
                                   config.sweeps_between_bins);
  meas_handler->set_observables(config.observables);

  // --------------------------------------------------------------------------------------------------
  //                                Parse the input Momentum parmas
  // --------------------------------------------------------------------------------------------------
  // make sure that the lattice module is initialized ahead
  if (lattice->InitialStatus()) {
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
  }
}

void Initializer::initial_modules(ModelBase& model, LatticeBase& lattice, Walker& walker,
                                  MeasureHandler& meas_handler) {
  // make sure that the module objects have been created,
  // and the parameters are setup correctly in advance.
  // notice that the orders of initializations below are important.

  // initialize lattice module
  if (!lattice.InitialStatus()) {
    lattice.initial();
  }

  // initialize MeasureHandler module
  meas_handler.initial(lattice, walker);

  // initialize dqmcWalker module
  walker.initial(lattice, meas_handler);

  // initialize model module
  // naively link
  model.initial(lattice, walker);
  model.link();
}

void Initializer::initial_modules(ModelBase& model, LatticeBase& lattice, Walker& walker,
                                  MeasureHandler& meas_handler, CheckerBoardBase& checkerboard) {
  // make sure that the module objects have been created,
  // and the parameters are setup correctly in advance.
  // notice that the orders of initializations below are important.

  // initialize lattice module
  if (!lattice.InitialStatus()) {
    lattice.initial();
  }

  // initialize MeasureHandler module
  meas_handler.initial(lattice, walker);

  // initialize dqmcWalker module
  walker.initial(lattice, meas_handler);

  // initialize model module
  model.initial(lattice, walker);

  // initialize checkerboard module and link to the model class
  checkerboard.set_checkerboard_params(lattice, model, walker);
  checkerboard.initial();
  model.link(checkerboard);
}

void Initializer::initial_dqmc(ModelBase& model, LatticeBase& lattice, Walker& walker,
                               [[maybe_unused]] MeasureHandler& meas_handler) {
  // this subroutine should be called after the initial
  // configuration of the bosonic fields have been determined,
  // either randomly initialized or read from a input config file.
  // SvdStack class are initialized and the greens functions
  // for the initial bosonic fields are computed in this function.
  walker.initial_svd_stacks(lattice, model);
  walker.initial_greens_functions();
  walker.initial_config_sign();
}

}  // namespace DQMC
