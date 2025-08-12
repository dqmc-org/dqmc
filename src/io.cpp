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

  if (!infile.is_open()) {
    throw std::runtime_error(
        "DQMC::IO::read_bosonic_fields_from_file(): fail to open file '" +
        filename + "'.");
  }

  try {
    model.read_auxiliary_field_from_stream(infile);
  } catch (const std::exception& e) {
    infile.close();
    throw std::runtime_error("DQMC::IO::read_bosonic_fields_from_file(): " +
                             std::string(e.what()));
  }

  infile.close();
}

void IO::output_bosonic_fields(std::ostream& ostream, const ModelBase& model) {
  if (!ostream) {
    throw std::runtime_error(
        "DQMC::IO::output_bosonic_fields(): output stream is not valid.");
  }
  model.output_configuration(ostream);
}

}  // namespace DQMC
