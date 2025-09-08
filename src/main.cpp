#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include "dqmc.h"
#include "utils/logger.h"

// the main program
int main(int argc, char* argv[]) {
  namespace po = boost::program_options;

  std::string config_file{};
  std::string out_path{};
  int seed = 0;

  auto& logger = Utils::Logger::the();

  // clang-format off
  po::options_description general("General options");
  general.add_options()
    ("help,h",                                                                      "Show help message")
    ("config,c",    po::value<std::string>(&config_file),                           "Configuration file path")
    ("output,o",    po::value<std::string>(&out_path)->default_value("../example"), "Output folder path")
    ("seed,s",      po::value<int>(&seed)->default_value(0),                        "Random seed for simulation")
    ("fields,f",    po::value<std::string>()->default_value(""),                    "Path to auxiliary fields configuration file");

  po::options_description model_opts("Model options");
  model_opts.add_options()
    ("model.type",               po::value<std::string>()->default_value("AttractiveHubbard"), "Model type (AttractiveHubbard, RepulsiveHubbard)")
    ("model.hopping_t",          po::value<double>()->default_value(1.0),                      "Hopping parameter t")
    ("model.onsite_u",           po::value<double>()->default_value(4.0),                      "On-site interaction U")
    ("model.chemical_potential", po::value<double>()->default_value(0.0),                      "Chemical potential μ");

  po::options_description lattice_opts("Lattice options");
  lattice_opts.add_options()
    ("lattice.type", po::value<std::string>()->default_value("Square"),                         "Lattice type (Square, Cubic, Chain, Honeycomb)")
    ("lattice.size", po::value<std::vector<int>>()->multitoken()->default_value({4, 4}, "4 4"), "Lattice dimensions");

  po::options_description checkerboard_opts("Checkerboard options");
  checkerboard_opts.add_options()
    ("checkerboard.enable", po::value<bool>()->default_value(false),                           "Enable checkerboard decomposition");

  po::options_description mc_opts("Monte Carlo options");
  mc_opts.add_options()
    ("mc.beta",               po::value<double>()->default_value(8.0),                         "Inverse temperature β")
    ("mc.time_size",          po::value<double>()->default_value(160),                         "Imaginary time discretization")
    ("mc.stabilization_pace", po::value<int>()->default_value(10),                             "QR decomposition frequency");

  po::options_description measure_opts("Measurement options (Adaptive Binning)");
  measure_opts.add_options()
    ("measure.sweeps_warmup",                   po::value<int>()->default_value(512),            "Warmup sweeps")
    ("measure.block_size",                      po::value<int>()->default_value(100),            "Number of sweeps per block for averaging")
    ("measure.autobinning_target_observable",  po::value<std::string>()->default_value("double_occupancy"), "Scalar observable to monitor for convergence")
    ("measure.autobinning_target_rel_error",   po::value<double>()->default_value(0.01),        "Target relative error for convergence (e.g., 0.01 for 1%)")
    ("measure.autobinning_max_sweeps",         po::value<int>()->default_value(2000000),        "Maximum measurement sweeps (safety cutoff)")
    ("measure.autobinning_min_sweeps",         po::value<int>()->default_value(20000),          "Minimum sweeps before convergence checks");

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
        logger.warn("Could not open config file: {}", config_file);
      }
    }
  } catch (const po::error& e) {
    logger.error("Command line parsing error: {}", e.what());
    return 1;
  }

  if (vm.count("help")) {
    std::cout << argv[0] << "\n" << cmdline_options << std::endl;
    return 0;
  }

  // Setup the directories
  const std::filesystem::path seeded_output_path = out_path + std::to_string(seed);
  const std::filesystem::path results_path = seeded_output_path / "results";
  const std::filesystem::path bins_path = seeded_output_path / "bins";

  std::filesystem::create_directories(seeded_output_path);
  std::filesystem::create_directories(results_path);
  std::filesystem::create_directories(bins_path);

  logger.set_file(seeded_output_path / "app.log");

  // ------------------------------------------------------------------------------------------------
  //                                Output current date and time
  // ------------------------------------------------------------------------------------------------
  const auto current_sys_time = std::chrono::system_clock::now();
  const auto local_zone_ptr = std::chrono::current_zone();
  const auto zoned_local_time = std::chrono::zoned_time(local_zone_ptr, current_sys_time);
  logger.info("Current time: {:%Y-%m-%d %H:%M:%S}", zoned_local_time);

  // ------------------------------------------------------------------------------------------------
  //                                Output run information
  // ------------------------------------------------------------------------------------------------
  logger.info("Starting DQMC run with seed: {}", seed);

  // ------------------------------------------------------------------------------------------------
  //                                 Process of DQMC simulation
  // ------------------------------------------------------------------------------------------------

  try {
    // 1. Create config object from parsed options
    DQMC::Config config{};
    config.seed = seed;
    config.fields_file = vm["fields"].as<std::string>();
    config.model_type = vm["model.type"].as<std::string>();
    config.hopping_t = vm["model.hopping_t"].as<double>();
    config.onsite_u = vm["model.onsite_u"].as<double>();
    config.chemical_potential = vm["model.chemical_potential"].as<double>();

    config.lattice_type = vm["lattice.type"].as<std::string>();
    config.lattice_size = vm["lattice.size"].as<std::vector<int>>();

    config.enable_checkerboard = vm["checkerboard.enable"].as<bool>();

    config.beta = vm["mc.beta"].as<double>();
    config.time_size = vm["mc.time_size"].as<double>();
    config.stabilization_pace = vm["mc.stabilization_pace"].as<int>();

    config.sweeps_warmup = vm["measure.sweeps_warmup"].as<int>();
    config.block_size = vm["measure.block_size"].as<int>();
    config.autobinning_target_observable =
        vm["measure.autobinning_target_observable"].as<std::string>();
    config.autobinning_target_rel_error = vm["measure.autobinning_target_rel_error"].as<double>();
    config.autobinning_max_sweeps = vm["measure.autobinning_max_sweeps"].as<int>();
    config.autobinning_min_sweeps = vm["measure.autobinning_min_sweeps"].as<int>();

    config.observables = vm["observables"].as<std::vector<std::string>>();

    config.momentum = vm["momentum.point"].as<std::string>();
    config.momentum_list = vm["momentum.list"].as<std::string>();

    // 2. Create the stateful Dqmc object. All initialization happens here.
    DQMC::Dqmc simulation(config);

    // 3. Output initialization info
    simulation.initial_message(std::cout);

    // 4. Run the entire simulation
    simulation.run();

    // 5. Output ending info
    simulation.info_message();

    // 6. Output scalar results to console
    simulation.output_results(std::cout);

    // 7. Write all detailed results to files
    simulation.write_results(results_path, bins_path);

  } catch (const std::exception& e) {
    logger.error("Simulation error: {}", e.what());
    return 1;
  }

  return 0;
}
