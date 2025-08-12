#include "lattice/square.h"

#include <format>

namespace Lattice {

// high symmetry points in the reciprocal lattice
const LatticeInt Square::GammaPointIndex() const {
  return this->m_gamma_point_index;
}
const LatticeInt Square::XPointIndex() const { return this->m_x_point_index; }
const LatticeInt Square::MPointIndex() const { return this->m_m_point_index; }
const LatticeIntVec& Square::DeltaLineIndex() const {
  return this->m_delta_line_index;
}
const LatticeIntVec& Square::ZLineIndex() const { return this->m_z_line_index; }
const LatticeIntVec& Square::SigmaLineIndex() const {
  return this->m_sigma_line_index;
}
const LatticeIntVec& Square::Gamma2X2M2GammaLoopIndex() const {
  return this->m_gamma2x2m2gamma_loop_index;
}

void Square::output_lattice_info(std::ostream& ostream,
                                 int momentum_index) const {
  auto fmt_param_str = [](const std::string& desc, const std::string& joiner,
                          const std::string& value) {
    return std::format("{:>30s}{:>7s}{:>24s}\n", desc, joiner, value);
  };
  std::string joiner = "->";

  auto fmt_cell = [](int side) { return std::format("{} * {}", side, side); };
  auto fmt_momentum = [](double px, double py) {
    return std::format("({:.2f}, {:.2f}) pi", px, py);
  };

  const double px = (this->Index2Momentum(momentum_index, 0) / M_PI);
  const double py = (this->Index2Momentum(momentum_index, 1) / M_PI);

  ostream << "   Lattice: Square lattice\n"
          << fmt_param_str("Size of cell", joiner,
                           fmt_cell(this->m_side_length))
          << fmt_param_str("Momentum point", joiner, fmt_momentum(px, py))
          << std::flush;
}

void Square::set_lattice_params(const LatticeIntVec& side_length_vec) {
  // lattice in two dimension
  assert((int)side_length_vec.size() == 2);
  // for square lattice, the length of each side should be equal to each other
  assert(side_length_vec[0] == side_length_vec[1]);
  assert(side_length_vec[0] >= 2);

  this->m_space_dim = 2;
  this->m_coordination_number = 4;
  this->m_side_length = side_length_vec[0];
  this->m_space_size = side_length_vec[0] * side_length_vec[1];
}

int Square::site_to_index(int x, int y) const {
  return x + this->m_side_length * y;
}

std::array<int, 2> Square::index_to_site(int index) const {
  return {index % this->m_side_length, index / this->m_side_length};
}

void Square::initial_index2site_table() {
  this->m_index2site_table.resize(this->m_space_size, this->m_space_dim);
  for (auto index = 0; index < this->m_space_size; ++index) {
    // map the site index to the site vector (x,y)
    auto [i, j] = index_to_site(index);
    this->m_index2site_table(index, 0) = i;
    this->m_index2site_table(index, 1) = j;
  }
}

void Square::initial_index2momentum_table() {
  // k stars (inequivalent momentum points) in 2d square lattice
  // locate in the zone surrounded by loop (0,0) -> (pi,0) -> (pi,pi) ->
  // (0,0). note that the point group of 2d sqaure lattice is C4v
  this->m_num_k_stars = (std::floor(this->m_side_length / 2.0) + 1) *
                        (std::floor(this->m_side_length / 2.0) + 2) / 2;

  // initialize indices of k stars
  this->m_k_stars_index.reserve(this->m_num_k_stars);
  for (auto index = 0; index < this->m_num_k_stars; ++index) {
    this->m_k_stars_index.emplace_back(index);
  }

  // initialize index2momentum table
  this->m_index2momentum_table.resize(this->m_num_k_stars, this->m_space_dim);
  int count = 0;
  for (auto i = std::ceil(this->m_side_length / 2.0); i <= this->m_side_length;
       ++i) {
    for (auto j = std::ceil(this->m_side_length / 2.0); j <= i; ++j) {
      this->m_index2momentum_table.row(count) =
          Eigen::Vector2d((double)i / this->m_side_length * 2 * M_PI - M_PI,
                          (double)j / this->m_side_length * 2 * M_PI - M_PI);
      count++;
    }
  }
}

void Square::initial_nearest_neighbour_table() {
  // the coordination number for 2d square lattice is 4
  // correspondense between the table index and the direction of displacement
  // : 0: (x+1, y)    1: (x, y+1) 2: (x-1, y)    3: (x, y-1)
  this->m_nearest_neighbour_table.resize(this->m_space_size,
                                         this->m_coordination_number);
  int L = this->m_side_length;
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      int site_index = this->site_to_index(i, j);

      // Direction 0: (x+1, y)
      this->m_nearest_neighbour_table(site_index, 0) =
          this->site_to_index((i + 1) % L, j);

      // Direction 1: (x, y+1)
      this->m_nearest_neighbour_table(site_index, 1) =
          this->site_to_index(i, (j + 1) % L);

      // Direction 2: (x-1, y)
      this->m_nearest_neighbour_table(site_index, 2) =
          this->site_to_index((i - 1 + L) % L, j);

      // Direction 3: (x, y-1)
      this->m_nearest_neighbour_table(site_index, 3) =
          this->site_to_index(i, (j - 1 + L) % L);
    }
  }
}

void Square::initial_displacement_table() {
  this->m_displacement_table.resize(this->m_space_size, this->m_space_size);
  int L = this->m_side_length;
  for (auto i = 0; i < this->m_space_size; ++i) {
    const auto [xi, yi] = index_to_site(i);
    for (auto j = 0; j < this->m_space_size; ++j) {
      const auto [xj, yj] = index_to_site(j);
      // displacement pointing from site i to site j
      const auto dx = (xj - xi + L) % L;
      const auto dy = (yj - yi + L) % L;
      this->m_displacement_table(i, j) = this->site_to_index(dx, dy);
    }
  }
}

void Square::initial_symmetry_points() {
  // high symmetry points of 2d square lattice
  // Gamma point:  (0,  0)
  // X point:      (pi, 0)
  // M point:      (pi, pi)
  this->m_gamma_point_index = 0;
  this->m_x_point_index =
      this->m_num_k_stars - std::floor(this->m_side_length / 2.0) - 1;
  this->m_m_point_index = this->m_num_k_stars - 1;

  // high symmetry lines of 2d square lattice
  // Delta line:   (0,0)  ->  (pi,0)
  // Z line:       (pi,0) ->  (pi,pi)
  // Sigma line:   (0,0)  ->  (pi,pi)
  this->m_delta_line_index.reserve(std::floor(this->m_side_length / 2.0) + 1);
  this->m_z_line_index.reserve(std::floor(this->m_side_length / 2.0) + 1);
  this->m_sigma_line_index.reserve(std::floor(this->m_side_length / 2.0) + 1);
  for (auto i = 0; i < std::floor(this->m_side_length / 2.0) + 1; ++i) {
    this->m_delta_line_index.emplace_back(i * (i + 1) / 2);
    this->m_z_line_index.emplace_back(this->m_x_point_index + i);
    this->m_sigma_line_index.emplace_back(i * (i + 3) / 2);
  }

  // loop: (0,0) -> (pi,0) -> (pi,pi) -> (0,0)
  this->m_gamma2x2m2gamma_loop_index.reserve(
      3 * (this->m_side_length - std::ceil(this->m_side_length / 2.0)));
  for (auto i = 0; i < std::floor(this->m_side_length / 2.0); ++i) {
    // along (0,0) -> (pi,0) direation
    this->m_gamma2x2m2gamma_loop_index.emplace_back(i * (i + 1) / 2);
  }
  for (auto i = 0; i < std::floor(this->m_side_length / 2.0); ++i) {
    // along (pi,0) -> (pi,pi) direction
    this->m_gamma2x2m2gamma_loop_index.emplace_back(this->m_x_point_index + i);
  }
  for (auto i = std::floor(this->m_side_length / 2.0); i >= 1; --i) {
    // along (pi,pi) -> (0,0) direction
    this->m_gamma2x2m2gamma_loop_index.emplace_back(i * (i + 3) / 2);
  }
}

void Square::initial_fourier_factor_table() {
  // Re( exp(-ikx) ) for lattice site x and momentum k
  this->m_fourier_factor_table.resize(this->m_space_size, this->m_num_k_stars);
  for (auto i = 0; i < this->m_space_size; ++i) {
    for (auto k = 0; k < this->m_num_k_stars; ++k) {
      // this defines the inner product of a site vector x and a momemtum
      // vector
      // k
      auto [xi, yi] = index_to_site(i);
      this->m_fourier_factor_table(i, k) =
          cos((-xi * this->m_index2momentum_table(k, 0) -
               yi * this->m_index2momentum_table(k, 1)));
    }
  }
}

void Square::initial_hopping_matrix() {
  this->m_hopping_matrix.resize(this->m_space_size, this->m_space_size);
  for (auto index = 0; index < this->m_space_size; ++index) {
    // direction 0 for x+1 and 1 for y+1
    const int index_xplus1 = this->NearestNeighbour(index, 0);
    const int index_yplus1 = this->NearestNeighbour(index, 1);

    this->m_hopping_matrix(index, index_xplus1) += 1.0;
    this->m_hopping_matrix(index_xplus1, index) += 1.0;
    this->m_hopping_matrix(index, index_yplus1) += 1.0;
    this->m_hopping_matrix(index_yplus1, index) += 1.0;
  }
}

void Square::initial() {
  // avoid multiple initialization
  if (!this->m_initial_status) {
    this->initial_index2site_table();
    this->initial_index2momentum_table();

    this->initial_nearest_neighbour_table();
    this->initial_displacement_table();
    this->initial_symmetry_points();
    this->initial_fourier_factor_table();

    this->initial_hopping_matrix();

    this->m_initial_status = true;
  }
}

}  // namespace Lattice
