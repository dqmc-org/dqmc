#include "io.h"

namespace DQMC {
void IO::output_init_info(std::ostream& ostream, int world_size,
                          const ModelBase& model, const LatticeBase& lattice,
                          const Walker& walker,
                          const MeasureHandler& meas_handler,
                          const CheckerBoardBasePtr& checkerboard) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_init_info(): output stream is not valid.");
  }

  auto fmt_param_str = [](const std::string& desc, const std::string& joiner,
                          const std::string& value) {
    return std::format("{:>30s}{:>7s}{:>24s}\n", desc, joiner, value);
  };

  model.output_model_info(ostream);
  lattice.output_lattice_info(ostream, meas_handler.Momentum());

  ostream << fmt_param_str("Checkerboard breakups", "->",
                           checkerboard ? "True" : "False")
          << std::endl;

  walker.output_montecarlo_info(ostream);
  meas_handler.output_measuring_info(ostream, world_size);
}

void IO::output_ending_info(std::ostream& ostream, const Walker& walker) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_ending_info(): output stream is not valid.");
  }

  auto duration = Dqmc::timer_as_duration();

  auto d = std::chrono::duration_cast<std::chrono::days>(duration);
  duration -= d;

  auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= h;

  auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= m;

  auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
  duration -= s;

  ostream << std::format(
      "\n>> The simulation finished in {}d {}h {}m {}s {}ms.\n", d.count(),
      h.count(), m.count(), s.count(), duration.count());

  ostream << std::format(">> Maximum of the wrapping error: {:.5e}\n",
                         walker.WrapError());
}

void IO::read_bosonic_fields_from_file(const std::string& filename,
                                       ModelBase& model) {
  std::ifstream infile(filename, std::ios::in);

  // check whether the ifstream works well
  if (!infile.is_open()) {
    std::cerr << "DQMC::IO::read_bosonic_fields_from_file(): "
              << "fail to open file \'" << filename << "\'." << std::endl;
    exit(1);
  }

  // temporary parameters
  std::string line;
  std::vector<std::string> data;

  // note that the IO class should be a friend class of any derived model
  // class to get access to the bosonic fields member

  // ---------------------------------  Repulsive Hubbard model
  // ------------------------------------
  if (auto repulsive_hubbard = dynamic_cast<Model::RepulsiveHubbard*>(&model);
      repulsive_hubbard != nullptr) {
    // consistency check of the model parameters
    // read the first line which containing the model information
    getline(infile, line);
    boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
    data.erase(std::remove(std::begin(data), std::end(data), ""),
               std::end(data));

    const int time_size = boost::lexical_cast<int>(data[0]);
    const int space_size = boost::lexical_cast<int>(data[1]);
    if ((time_size != repulsive_hubbard->m_bosonic_field.rows()) ||
        (space_size != repulsive_hubbard->m_bosonic_field.cols())) {
      std::cerr
          << "DQMC::IO::read_bosonic_fields_from_file(): "
          << "inconsistency between model settings and input configs (time or "
             "space size). "
          << std::endl;
      exit(1);
    }

    // read in the configurations of auxiliary fields
    int time_point, space_point;
    while (getline(infile, line)) {
      boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
      data.erase(std::remove(std::begin(data), std::end(data), ""),
                 std::end(data));
      time_point = boost::lexical_cast<int>(data[0]);
      space_point = boost::lexical_cast<int>(data[1]);
      repulsive_hubbard->m_bosonic_field(time_point, space_point) =
          boost::lexical_cast<double>(data[2]);
    }
    // close the file stream
    infile.close();
  }

  // ---------------------------------  Attractive Hubbard model
  // -----------------------------------
  else if (auto attractive_hubbard =
               dynamic_cast<Model::AttractiveHubbard*>(&model);
           attractive_hubbard != nullptr) {
    // consistency check of the model parameters
    // read the first line which containing the model information
    getline(infile, line);
    boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
    data.erase(std::remove(std::begin(data), std::end(data), ""),
               std::end(data));

    const int time_size = boost::lexical_cast<int>(data[0]);
    const int space_size = boost::lexical_cast<int>(data[1]);
    if ((time_size != attractive_hubbard->m_bosonic_field.rows()) ||
        (space_size != attractive_hubbard->m_bosonic_field.cols())) {
      std::cerr
          << "DQMC::IO::read_bosonic_fields_from_file(): "
          << "inconsistency between model settings and input configs (time or "
             "space size). "
          << std::endl;
      exit(1);
    }

    // read in the configurations of auxiliary fields
    int time_point, space_point;
    while (getline(infile, line)) {
      boost::split(data, line, boost::is_any_of(" "), boost::token_compress_on);
      data.erase(std::remove(std::begin(data), std::end(data), ""),
                 std::end(data));
      time_point = boost::lexical_cast<int>(data[0]);
      space_point = boost::lexical_cast<int>(data[1]);
      attractive_hubbard->m_bosonic_field(time_point, space_point) =
          boost::lexical_cast<double>(data[2]);
    }
    // close the file stream
    infile.close();
  }

  // other model types, raising errors
  else {
    // close the file stream
    infile.close();
    std::cerr << "DQMC::IO::read_bosonic_fields_from_file(): "
              << "undefined model type." << std::endl;
    exit(1);
  }
}

void IO::output_bosonic_fields(std::ostream& ostream, const ModelBase& model) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_bosonic_fields(): output stream is not valid.");
  }
  model.output_configuration(ostream);
}

}  // namespace DQMC
