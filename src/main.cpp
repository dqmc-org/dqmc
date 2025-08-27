#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
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
  namespace po = boost::program_options;

  std::string config_file{};
  std::string fields_file{};
  std::string out_path{};
  int run_id = 0;

  // clang-format off
  po::options_description general("General options");
  general.add_options()
    ("help,h",                                                                      "Show help message")
    ("config,c",    po::value<std::string>(&config_file),                           "Configuration file path")
    ("output,o",    po::value<std::string>(&out_path)->default_value("../example"), "Output folder path")
    ("fields,f",    po::value<std::string>(&fields_file),                           "Path to auxiliary fields configuration file")
    ("run-id,r",    po::value<int>(&run_id)->default_value(0),                      "Unique run identifier for parallel execution");

  po::options_description model_opts("Model options");
  model_opts.add_options()
    ("model.type",               po::value<std::string>()->default_value("AttractiveHubbard"), "Model type (AttractiveHubbard, RepulsiveHubbard)")
    ("model.hopping_t",          po::value<double>()->default_value(1.0),                      "Hopping parameter t")
    ("model.onsite_u",           po::value<double>()->default_value(4.0),                      "On-site interaction U")
    ("model.chemical_potential", po::value<double>()->default_value(0.0),                      "Chemical potential μ");

  po::options_description lattice_opts("Lattice options");
  lattice_opts.add_options()
    ("lattice.type", po::value<std::string>()->default_value("Square"),                         "Lattice type (Square, Cubic, Honeycomb)")
    ("lattice.size", po::value<std::vector<int>>()->multitoken()->default_value({4, 4}, "4 4"), "Lattice dimensions");

  po::options_description checkerboard_opts("Checkerboard options");
  checkerboard_opts.add_options()
    ("checkerboard.enable", po::value<bool>()->default_value(false),                           "Enable checkerboard decomposition");

  po::options_description mc_opts("Monte Carlo options");
  mc_opts.add_options()
    ("mc.beta",               po::value<double>()->default_value(8.0),                         "Inverse temperature β")
    ("mc.time_size",          po::value<double>()->default_value(160),                         "Imaginary time discretization")
    ("mc.stabilization_pace", po::value<int>()->default_value(10),                             "QR decomposition frequency");

  po::options_description measure_opts("Measurement options");
  measure_opts.add_options()
    ("measure.sweeps_warmup",       po::value<int>()->default_value(512),                      "Warmup sweeps")
    ("measure.bin_num",             po::value<int>()->default_value(20),                       "Number of bins")
    ("measure.bin_size",            po::value<int>()->default_value(100),                      "Bin size")
    ("measure.sweeps_between_bins", po::value<int>()->default_value(20),                       "Sweeps between bins");

  po::options_description observable_opts("Observable options");
  observable_opts.add_options()
    ("observables", po::value<std::vector<std::string>>()->multitoken()->default_value(
          {"filling_number",
           "double_occupancy",
           "kinetic_energy",
           "momentum_distribution",
           "local_spin_corr",
           "spin_density_structure_factor",
           "charge_density_structure_factor",
           "s_wave_pairing_corr",
           "greens_functions",
           "density_of_states",
           "superfluid_stiffness",
           "dynamic_spin_susceptibility"},
          "filling_number double_occupancy ..."),                                               "List of observables to measure");

  po::options_description momentum_opts("Momentum options");
  momentum_opts.add_options()
    ("momentum.point", po::value<std::string>()->default_value("MPoint"),                      "Momentum point (GammaPoint, MPoint, XPoint, RPoint)")
    ("momentum.list",  po::value<std::string>()->default_value("KstarsAll"),                   "Momentum list (KstarsAll, DeltaLine, ZLine, etc.)");
  // clang-format on

  po::options_description cmdline_options("All options");
  cmdline_options.add(general)
      .add(model_opts)
      .add(lattice_opts)
      .add(checkerboard_opts)
      .add(mc_opts)
      .add(measure_opts)
      .add(observable_opts)
      .add(momentum_opts);

  po::options_description config_file_options;
  config_file_options.add(model_opts)
      .add(lattice_opts)
      .add(checkerboard_opts)
      .add(mc_opts)
      .add(measure_opts)
      .add(observable_opts)
      .add(momentum_opts);

  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);

    if (vm.count("config")) {
      std::ifstream ifs(config_file);
      if (ifs) {
        po::store(po::parse_config_file(ifs, config_file_options), vm);
        po::notify(vm);
      } else {
        std::cerr << "Warning: Could not open config file: " << config_file << std::endl;
      }
    }
  } catch (const po::error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  if (vm.count("help")) {
    std::cout << argv[0] << "\n" << cmdline_options << std::endl;
    return 0;
  }

  try {
    std::filesystem::create_directories(out_path);
  } catch (const std::filesystem::filesystem_error& e) {
    throw std::runtime_error(
        std::format("DQMC::main(): Failed to initialize output folder at {}.", out_path));
  }

  // ------------------------------------------------------------------------------------------------
  //                                Output current date and time
  // ------------------------------------------------------------------------------------------------
  const auto current_sys_time = std::chrono::system_clock::now();
  const auto local_zone_ptr = std::chrono::current_zone();
  const auto zoned_local_time = std::chrono::zoned_time(local_zone_ptr, current_sys_time);
  std::cout << std::format(">> Current time: {:%Y-%m-%d %H:%M:%S}\n", zoned_local_time);

  // ------------------------------------------------------------------------------------------------
  //                                Output run information
  // ------------------------------------------------------------------------------------------------
  std::cout << std::format(">> Starting DQMC run with ID: {}\n", run_id) << std::endl;

  // ------------------------------------------------------------------------------------------------
  //                                 Process of DQMC simulation
  // ------------------------------------------------------------------------------------------------

  // set up random seeds for different runs
  // Use run_id for random seed variation to ensure different parallel runs use
  // different seeds
  std::default_random_engine rng(42 + run_id);

  // -----------------------------------  Initializations
  // ------------------------------------------

  // Create config from parsed options
  DQMC::Initializer::Config config{};

  // Populate config from program options
  config.model_type = std::move(vm["model.type"].as<std::string>());
  config.hopping_t = vm["model.hopping_t"].as<double>();
  config.onsite_u = vm["model.onsite_u"].as<double>();
  config.chemical_potential = vm["model.chemical_potential"].as<double>();

  config.lattice_type = std::move(vm["lattice.type"].as<std::string>());
  config.lattice_size = vm["lattice.size"].as<std::vector<int>>();

  config.enable_checkerboard = vm["checkerboard.enable"].as<bool>();

  config.beta = vm["mc.beta"].as<double>();
  config.time_size = vm["mc.time_size"].as<double>();
  config.stabilization_pace = vm["mc.stabilization_pace"].as<int>();

  config.sweeps_warmup = vm["measure.sweeps_warmup"].as<int>();
  config.bin_num = vm["measure.bin_num"].as<int>();
  config.bin_size = vm["measure.bin_size"].as<int>();
  config.sweeps_between_bins = vm["measure.sweeps_between_bins"].as<int>();

  config.observables = std::move(vm["observables"].as<std::vector<std::string>>());

  config.momentum = std::move(vm["momentum.point"].as<std::string>());
  config.momentum_list = std::move(vm["momentum.list"].as<std::string>());

  // Create and own all modules by parsing the config.
  auto context = DQMC::Initializer::parse_config(config);

  // Initialize modules using the context.
  DQMC::Initializer::initial_modules(context);

  if (fields_file.empty()) {
    // randomly initialize the bosonic fields if there are no input field configs
    context.model->set_bosonic_fields_to_random(rng);
    std::cout << ">> Configurations of the bosonic fields set to random.\n" << std::endl;
  } else {
    DQMC::IO::read_bosonic_fields_from_file(fields_file, *context.model);
    std::cout << ">> Configurations of the bosonic fields read from the "
                 "input config file.\n"
              << std::endl;
  }

  // initialize dqmc, preparing for the simulation
  DQMC::Initializer::initial_dqmc(context);

  std::cout << ">> Initialization finished. \n\n"
            << ">> The simulation is going to get started with parameters "
               "shown below :\n"
            << std::endl;

  // Create the simulation object, which takes ownership of the context
  DQMC::Dqmc simulation(std::move(context));

  // output the initialization info
  DQMC::IO::output_init_info(std::cout, simulation);

  // set up progress bar
  simulation.show_progress_bar(true);
  simulation.progress_bar_format(60, '=', ' ');
  simulation.set_refresh_rate(10);

  // ---------------------------------  Crucial simulation steps
  // ------------------------------------

  // the dqmc simulation start
  simulation.timer_begin();
  simulation.thermalize(rng);
  simulation.measure(rng);
  simulation.analyse();
  simulation.timer_end();

  // output the ending info
  DQMC::IO::output_ending_info(std::cout, simulation.walker(), simulation.timer_as_duration());

  // ---------------------------------  Output measuring results
  // ------------------------------------

  // screen output the results of scalar observables
  for (const auto& obs_name : simulation.handler().ObservablesList()) {
    if (auto obs = simulation.handler().find<Observable::Scalar>(obs_name)) {
      DQMC::IO::output_observable_to_console(std::cout, *obs);
    }
  }

  // file output
  std::ofstream outfile;

  // output the configurations of the bosonic fields
  const auto fields_out = (fields_file.empty())
                              ? std::format("{}/bosonic_fields_{}.out", out_path, run_id)
                              : fields_file;
  outfile.open(fields_out, std::ios::trunc);
  DQMC::IO::output_bosonic_fields(outfile, simulation.model());
  outfile.close();

  // output the k stars
  outfile.open(std::format("{}/kstars.out", out_path), std::ios::trunc);
  DQMC::IO::output_k_stars(outfile, simulation.lattice());
  outfile.close();

  // output the imaginary-time grids
  outfile.open(std::format("{}/imaginary_time_grids.out", out_path), std::ios::trunc);
  DQMC::IO::output_imaginary_time_grids(outfile, simulation.walker());
  outfile.close();

  // output measuring results of the observables

  // helper lambda to output observable to files
  auto output_observable_files = [&](const auto& obs, const std::string& obs_name) {
    // output of means and errors
    outfile.open(std::format("{}/{}_{}.out", out_path, obs_name, run_id), std::ios::trunc);
    DQMC::IO::output_observable_to_file(outfile, *obs);
    outfile.close();

    // output of raw data in terms of bins
    outfile.open(std::format("{}/{}.bins.out", out_path, obs_name), std::ios::trunc);
    DQMC::IO::output_observable_in_bins_to_file(outfile, *obs);
    outfile.close();
  };

  // iterate through all observables and output them
  for (const auto& obs_name : simulation.handler().ObservablesList()) {
    if (auto obs = simulation.handler().find<Observable::Scalar>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = simulation.handler().find<Observable::Vector>(obs_name)) {
      output_observable_files(obs, obs_name);
    } else if (auto obs = simulation.handler().find<Observable::Matrix>(obs_name)) {
      output_observable_files(obs, obs_name);
    }
  }

  return 0;
}
