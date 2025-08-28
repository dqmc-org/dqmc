#include "io.h"

namespace DQMC {
void IO::output_init_info(std::ostream& ostream, const Dqmc& simulation) {
  if (!ostream) {
    throw std::runtime_error("DQMC::IO::output_init_info(): output stream is not valid.");
  }

  simulation.model().output_model_info(ostream);
  simulation.lattice().output_lattice_info(ostream, simulation.handler().Momentum());

  ostream << std::format("{:>30s}{:>7s}{:>24s}\n\n", "Checkerboard breakups", "->",
                         simulation.checkerboard() ? "True" : "False");

  simulation.walker().output_montecarlo_info(ostream);
  simulation.handler().output_measuring_info(ostream);
}

void IO::output_ending_info(std::ostream& ostream, const Walker& walker,
                            std::chrono::milliseconds duration) {
  if (!ostream) {
    throw std::runtime_error("DQMC::IO::output_ending_info(): output stream is not valid.");
  }

  auto d = std::chrono::duration_cast<std::chrono::days>(duration);
  duration -= d;

  auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= h;

  auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= m;

  auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
  duration -= s;

  ostream << std::format("\n>> The simulation finished in {}d {}h {}m {}s {}ms.\n", d.count(),
                         h.count(), m.count(), s.count(), duration.count());

  ostream << std::format(">> Maximum of the wrapping error: {:.5e}\n", walker.WrapError());
}
}  // namespace DQMC
