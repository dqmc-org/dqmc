#include "lattice/chain.h"

#include <cmath>
#include <format>

#include "utils/format_output.h"

namespace Lattice {

Chain::Chain(const std::vector<int>& lattice_size) {
  set_lattice_params(lattice_size);
  initial();
}

void Chain::output_lattice_info(std::ostream& ostream, int momentum_index) const {
  auto fmt_cell = [](int side) { return std::format("{}", side); };

  auto fmt_momentum = [](double px) { return std::format("({:.2f}) pi", px); };

  const double px = (this->index_to_momentum(momentum_index, 0) / M_PI);

  ostream << "   Lattice: 1D Chain\n"
          << Utils::FormatOutput::display("Size of cell", fmt_cell(this->m_side_length))
          << Utils::FormatOutput::display("Momentum point", fmt_momentum(px)) << std::flush;
}

void Chain::set_lattice_params(const std::vector<int>& side_length_vec) {
  // lattice in one dimension
  DQMC_ASSERT((int)side_length_vec.size() == 1);
  DQMC_ASSERT(side_length_vec[0] >= 2);

  this->m_space_dim = 1;
  this->m_coordination_number = 2;
  this->m_side_length = side_length_vec[0];
  this->m_space_size = side_length_vec[0];
}

int Chain::site_to_index(int x) const { return x; }

int Chain::index_to_site(int index) const { return index; }

void Chain::initial_index_to_site_table() {
  this->m_index_to_site_table.resize(this->m_space_size, this->m_space_dim);
  for (auto index = 0; index < this->m_space_size; ++index) {
    // map the site index to the site vector (x)
    this->m_index_to_site_table(index, 0) = index;
  }
}

void Chain::initial_index_to_momentum_table() {
  // k stars (inequivalent momentum points) in 1D chain.
  // Due to k <-> -k symmetry, we only need to consider k from 0 to pi.
  // Momenta are k_n = 2 * pi * n / L, for n = 0, 1, ..., L-1.
  // We need n such that 0 <= 2 * pi * n / L <= pi => 0 <= n <= L/2.
  this->m_num_k_stars = std::floor(this->m_side_length / 2.0) + 1;

  // initialize indices of k stars
  this->m_k_stars_index.reserve(this->m_num_k_stars);
  for (auto index = 0; index < this->m_num_k_stars; ++index) {
    this->m_k_stars_index.emplace_back(index);
  }

  // initialize index_to_momentum table
  this->m_index_to_momentum_table.resize(this->m_num_k_stars, this->m_space_dim);
  for (int i = 0; i < this->m_num_k_stars; ++i) {
    this->m_index_to_momentum_table(i, 0) = (double)i / this->m_side_length * 2.0 * M_PI;
  }
}

void Chain::initial_nearest_neighbor_table() {
  // the coordination number for 1D chain is 2
  // correspondence between the table index and the direction of displacement:
  // 0: (x+1) right
  // 1: (x-1) left
  this->m_nearest_neighbor_table.resize(this->m_space_size, this->m_coordination_number);
  int L = this->m_side_length;
  for (int i = 0; i < L; ++i) {
    int site_index = this->site_to_index(i);

    // Direction 0: (x+1)
    this->m_nearest_neighbor_table(site_index, 0) = this->site_to_index((i + 1) % L);

    // Direction 1: (x-1)
    this->m_nearest_neighbor_table(site_index, 1) = this->site_to_index((i - 1 + L) % L);
  }
}

void Chain::initial_displacement_table() {
  this->m_displacement_table.resize(this->m_space_size, this->m_space_size);
  int L = this->m_side_length;
  for (auto i = 0; i < this->m_space_size; ++i) {
    for (auto j = 0; j < this->m_space_size; ++j) {
      // displacement pointing from site i to site j
      const auto dx = (j - i + L) % L;
      this->m_displacement_table(i, j) = this->site_to_index(dx);
    }
  }
}

void Chain::initial_symmetry_points() {
  // high symmetry points of 1D chain
  // Gamma point:  k = 0
  // X point:      k = pi
  this->m_gamma_point_index = 0;
  this->m_x_point_index = std::floor(this->m_side_length / 2.0);

  // high symmetry line of 1D chain (all k-stars)
  // Delta line:   0 -> pi
  this->m_delta_line_index.reserve(this->m_num_k_stars);
  for (auto i = 0; i < this->m_num_k_stars; ++i) {
    this->m_delta_line_index.emplace_back(i);
  }

  m_momentum_points["GammaPoint"] = m_gamma_point_index;
  m_momentum_points["XPoint"] = m_x_point_index;

  m_momentum_lists["KstarsAll"] = m_k_stars_index;
  m_momentum_lists["DeltaLine"] = m_delta_line_index;
}

void Chain::initial_fourier_factor_table() {
  // Re( exp(-ikx) ) for lattice site x and momentum k
  this->m_fourier_factor_table.resize(this->m_space_size, this->m_num_k_stars);
  for (auto i = 0; i < this->m_space_size; ++i) {
    for (auto k = 0; k < this->m_num_k_stars; ++k) {
      // this defines the inner product of a site vector x and a momemtum vector k
      const int xi = this->index_to_site(i);
      const double kx = this->index_to_momentum(k, 0);
      this->m_fourier_factor_table(i, k) = cos(-xi * kx);
    }
  }
}

void Chain::initial_hopping_matrix() {
  this->m_hopping_matrix.resize(this->m_space_size, this->m_space_size);
  this->m_hopping_matrix.setZero();
  for (auto index = 0; index < this->m_space_size; ++index) {
    // direction 0 for x+1
    const int index_xplus1 = this->nearest_neighbor(index, 0);

    this->m_hopping_matrix(index, index_xplus1) += 1.0;
    this->m_hopping_matrix(index_xplus1, index) += 1.0;
  }
}

void Chain::initial() {
  this->initial_index_to_site_table();
  this->initial_index_to_momentum_table();

  this->initial_nearest_neighbor_table();
  this->initial_displacement_table();
  this->initial_symmetry_points();
  this->initial_fourier_factor_table();

  this->initial_hopping_matrix();
}

}  // namespace Lattice
