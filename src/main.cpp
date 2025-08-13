#include <boost/program_options.hpp>
#include <chrono>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "checkerboard/checkerboard_base.h"
#include "dqmc.h"
#include "initializer.h"
#include "io.h"
#include "lattice/lattice_base.h"
#include "measure/measure_handler.h"
#include "model/model_base.h"
#include "svd_stack.h"
#include "walker.h"

// the main program
int main(int argc, char* argv[]) {
  // ------------------------------------------------------------------------------------------------
  //                                      Program options
  // ------------------------------------------------------------------------------------------------

  std::string config_file{};
  std::string fields_file{};
  std::string out_path{};
  int run_id = 0;

  // read parameters from the command line
  boost::program_options::options_description opts("Program options");
  boost::program_options::variables_map vm;

  opts.add_options()("help,h", "display this information")(
      "config,c",
      boost::program_options::value<std::string>(&config_file)
          ->default_value("../example/config.toml"),
      "path of the configuration file, default: ../example/config.toml")(
      "output,o",
      boost::program_options::value<std::string>(&out_path)->default_value(
          "../example"),
      "folder path which stores the output of measuring results, default: "
      "../example")("fields,f",
                    boost::program_options::value<std::string>(&fields_file),
                    "path of the configurations of auxiliary fields, if not "
                    "assigned the fields are to be set randomly.")(
      "run-id,r", boost::program_options::value<int>(&run_id)->default_value(0),
      "unique run identifier for parallel execution");

  // parse the command line options
  try {
    boost::program_options::store(parse_command_line(argc, argv, opts), vm);
  } catch (...) {
    std::cerr << "main(): undefined options got from command line."
              << std::endl;
    exit(1);
  }
  boost::program_options::notify(vm);

  // show the helping messages
  if (vm.count("help")) {
    std::cerr << argv[0] << "\n" << opts << std::endl;
    return 0;
  }

  // initialize the output folder, create if not exist
  if (access(out_path.c_str(), 0) != 0) {
    const std::string command = "mkdir -p " + out_path;
    if (system(command.c_str()) != 0) {
      std::cerr << std::format("main(): fail to creat folder at {} .\n",
                               out_path)
                << std::endl;
      exit(1);
    }
  }

  // ------------------------------------------------------------------------------------------------
  //                                Output current date and time
  // ------------------------------------------------------------------------------------------------
  const auto current_time = std::chrono::system_clock::now();
  const auto time_t = std::chrono::system_clock::to_time_t(current_time);
  const auto local_time = std::localtime(&time_t);
  std::ostringstream time_stream;
  time_stream << std::put_time(local_time, "%Y-%m-%d %H:%M:%S");
  std::cout << std::format(">> Current time: {}\n", time_stream.str())
            << std::endl;

  // ------------------------------------------------------------------------------------------------
  //                                Output run information
  // ------------------------------------------------------------------------------------------------
  std::cout << std::format(">> Starting DQMC run with ID: {}\n", run_id)
            << std::endl;

  // ------------------------------------------------------------------------------------------------
  //                                 Process of DQMC simulation
  // ------------------------------------------------------------------------------------------------

  // set up random seeds for different runs
  // Use run_id for random seed variation to ensure different parallel runs use
  // different seeds
  std::default_random_engine rng(42 + run_id);

  // -----------------------------------  Initializations
  // ------------------------------------------

  // create dqmc module objects
  std::unique_ptr<Model::ModelBase> model;
  std::unique_ptr<Lattice::LatticeBase> lattice;
  std::unique_ptr<DQMC::Walker> walker;
  std::unique_ptr<Measure::MeasureHandler> meas_handler;
  std::unique_ptr<CheckerBoard::CheckerBoardBase> checkerboard;

  // parse parmas from the configuation file
  DQMC::Initializer::parse_toml_config(config_file, 1, model, lattice, walker,
                                       meas_handler, checkerboard);

  // initialize modules
  if (checkerboard) {
    // using checkerboard break-up
    DQMC::Initializer::initial_modules(*model, *lattice, *walker, *meas_handler,
                                       *checkerboard);
  } else {
    // without checkerboard break-up
    DQMC::Initializer::initial_modules(*model, *lattice, *walker,
                                       *meas_handler);
  }

  if (fields_file.empty()) {
    // randomly initialize the bosonic fields if there are no input field
    // configs
    model->set_bosonic_fields_to_random(rng);
    std::cout << ">> Configurations of the bosonic fields set to random.\n"
              << std::endl;
  } else {
    DQMC::IO::read_bosonic_fields_from_file(fields_file, *model);
    std::cout << ">> Configurations of the bosonic fields read from the "
                 "input config file.\n"
              << std::endl;
  }

  // initialize dqmc, preparing for the simulation
  DQMC::Initializer::initial_dqmc(*model, *lattice, *walker, *meas_handler);

  std::cout << ">> Initialization finished. \n\n"
            << ">> The simulation is going to get started with parameters "
               "shown below :\n"
            << std::endl;

  // output the initialization info
  DQMC::IO::output_init_info(std::cout, 1, *model, *lattice, *walker,
                             *meas_handler, checkerboard);

  // set up progress bar
  DQMC::Dqmc::show_progress_bar(true);
  DQMC::Dqmc::progress_bar_format(60, '=', ' ');
  DQMC::Dqmc::set_refresh_rate(10);

  // ---------------------------------  Crucial simulation steps
  // ------------------------------------

  // the dqmc simulation start
  DQMC::Dqmc::timer_begin();
  DQMC::Dqmc::thermalize(*walker, *model, *lattice, *meas_handler, rng);
  DQMC::Dqmc::measure(*walker, *model, *lattice, *meas_handler, rng);

  // perform the analysis
  DQMC::Dqmc::analyse(*meas_handler);

  // end the timer
  DQMC::Dqmc::timer_end();

  // output the ending info
  DQMC::IO::output_ending_info(std::cout, *walker);

  // ---------------------------------  Output measuring results
  // ------------------------------------

  // screen output the results of scalar observables
  if (auto obs = meas_handler->find<Observable::ScalarObs>("equaltime_sign")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs = meas_handler->find<Observable::ScalarObs>("dynamic_sign")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  std::cout << std::endl;

  if (auto obs = meas_handler->find<Observable::ScalarObs>("filling_number")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("double_occupancy")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs = meas_handler->find<Observable::ScalarObs>("kinetic_energy")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs = meas_handler->find<Observable::ScalarObs>("local_spin_corr")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("momentum_distribution")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs = meas_handler->find<Observable::ScalarObs>(
          "spin_density_structure_factor")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs = meas_handler->find<Observable::ScalarObs>(
          "charge_density_structure_factor")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("s_wave_pairing_corr")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("superfluid_stiffness")) {
    DQMC::IO::output_observable_to_console(std::cout, *obs);
  }

  // file output
  std::ofstream outfile;

  // output the configurations of the bosonic fields
  // if there exist input file of fields configs, overwrite it.
  // otherwise the field configs are stored under the output folder.
  const auto fields_out =
      (fields_file.empty())
          ? out_path + "/fields_" + std::to_string(run_id) + ".out"
          : fields_file;
  outfile.open(fields_out, std::ios::trunc);
  DQMC::IO::output_bosonic_fields(outfile, *model);
  outfile.close();

  // output the k stars
  outfile.open(out_path + "/kstars.out", std::ios::trunc);
  DQMC::IO::output_k_stars(outfile, *lattice);
  outfile.close();

  // output the imaginary-time grids
  outfile.open(out_path + "/tgrids.out", std::ios::trunc);
  DQMC::IO::output_imaginary_time_grids(outfile, *walker);
  outfile.close();

  // output measuring results of the observables

  // s wave pairing correlation functions
  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("s_wave_pairing_corr")) {
    // output of means and errors
    outfile.open(out_path + "/swave_" + std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/swave.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // charge density structure factor
  if (auto obs = meas_handler->find<Observable::ScalarObs>(
          "charge_density_structure_factor")) {
    // output of means and errors
    outfile.open(out_path + "/charge_density_structure_factor_" +
                     std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/charge_density_structure_factor.bins.out",
                 std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // spin density structure factor
  if (auto obs = meas_handler->find<Observable::ScalarObs>(
          "spin_density_structure_factor")) {
    // output of means and errors
    outfile.open(out_path + "/spin_density_structure_factor_" +
                     std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/spin_density_structure_factor.bins.out",
                 std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // density of states
  if (auto obs =
          meas_handler->find<Observable::VectorObs>("density_of_states")) {
    // output of means and errors
    outfile.open(out_path + "/dos_" + std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/dos.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // equal-time sign
  if (auto obs = meas_handler->find<Observable::ScalarObs>("equaltime_sign")) {
    // output of means and errors
    outfile.open(
        out_path + "/equaltime_sign_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/equaltime_sign.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // dynamic sign
  if (auto obs = meas_handler->find<Observable::ScalarObs>("dynamic_sign")) {
    // output of means and errors
    outfile.open(out_path + "/dynamic_sign_" + std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/dynamic_sign.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // filling number
  if (auto obs = meas_handler->find<Observable::ScalarObs>("filling_number")) {
    // output of means and errors
    outfile.open(
        out_path + "/filling_number_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/filling_number.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // double occupancy
  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("double_occupancy")) {
    // output of means and errors
    outfile.open(
        out_path + "/double_occupancy_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/double_occupancy.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // kinetic energy
  if (auto obs = meas_handler->find<Observable::ScalarObs>("kinetic_energy")) {
    // output of means and errors
    outfile.open(
        out_path + "/kinetic_energy_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/kinetic_energy.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // local spin correlation
  if (auto obs = meas_handler->find<Observable::ScalarObs>("local_spin_corr")) {
    // output of means and errors
    outfile.open(
        out_path + "/local_spin_corr_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/local_spin_corr.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // momentum distribution
  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("momentum_distribution")) {
    // output of means and errors
    outfile.open(
        out_path + "/momentum_distribution_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/momentum_distribution.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // superfluid stiffness
  if (auto obs =
          meas_handler->find<Observable::ScalarObs>("superfluid_stiffness")) {
    // output of means and errors
    outfile.open(
        out_path + "/superfluid_stiffness_" + std::to_string(run_id) + ".out",
        std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/superfluid_stiffness.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // dynamical green's function in the reciprocal space
  if (auto obs =
          meas_handler->find<Observable::MatrixObs>("greens_functions")) {
    // output of means and errors
    outfile.open(out_path + "/greens_" + std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/greens.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  // dynamic spin susceptibility
  if (auto obs = meas_handler->find<Observable::VectorObs>(
          "dynamic_spin_susceptibility")) {
    // output of means and errors
    outfile.open(out_path + "/dss_" + std::to_string(run_id) + ".out",
                 std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(out_path + "/dss.bins.out", std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  }

  return 0;
}
