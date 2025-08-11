#include "lattice/cubic.h"

namespace Lattice {

// high symmetry points in the reciprocal lattice
const LatticeInt Cubic::GammaPointIndex() const {
  return this->m_gamma_point_index;
}
const LatticeInt Cubic::XPointIndex() const { return this->m_x_point_index; }
const LatticeInt Cubic::MPointIndex() const { return this->m_m_point_index; }
const LatticeInt Cubic::RPointIndex() const { return this->m_r_point_index; }
const LatticeIntVec& Cubic::DeltaLineIndex() const {
  return this->m_delta_line_index;
}
const LatticeIntVec& Cubic::ZLineIndex() const { return this->m_z_line_index; }
const LatticeIntVec& Cubic::SigmaLineIndex() const {
  return this->m_sigma_line_index;
}
const LatticeIntVec& Cubic::LambdaLineIndex() const {
  return this->m_lambda_line_index;
}
const LatticeIntVec& Cubic::SLineIndex() const { return this->m_s_line_index; }
const LatticeIntVec& Cubic::TLineIndex() const { return this->m_t_line_index; }

void Cubic::set_lattice_params(const LatticeIntVec& side_length_vec) {
  // lattice in three dimension
  assert((int)side_length_vec.size() == 3);
  // for cubic lattice, the length of each side should be equal to each other
  assert((side_length_vec[0] == side_length_vec[1]) &&
         (side_length_vec[0] == side_length_vec[2]));
  assert(side_length_vec[0] >= 2);

  this->m_space_dim = 3;
  this->m_coordination_number = 6;
  this->m_side_length = side_length_vec[0];
  this->m_space_size =
      side_length_vec[0] * side_length_vec[1] * side_length_vec[2];
}

void Cubic::initial_site_indexer() {
  // Initialize indexer with 3D dimensions for cubic lattice
  std::vector<int> dimensions = {this->m_side_length, this->m_side_length,
                                 this->m_side_length};
  this->m_site_indexer = Indexer(dimensions);
}

void Cubic::initial_index2momentum_table() {
  // the k stars (inequivalent momentum points) in 3d cubic lattice
  // locate in the trirectangular tetrahedron determined by the four vertices
  // (0,0,0), (pi,0,0), (pi,pi,0) and (pi,pi,pi).
  // note that the point group of 3d cubic lattice is Oh
  const auto half_side_length = std::floor(this->m_side_length / 2.0) + 1;
  this->m_num_k_stars = half_side_length * (half_side_length + 1) *
                            (2 * half_side_length + 1) / 12 +
                        half_side_length * (half_side_length + 1) / 4;

  // initialize indices of k stars
  this->m_k_stars_index.reserve(this->m_num_k_stars);
  for (auto index = 0; index < this->m_num_k_stars; ++index) {
    this->m_k_stars_index.emplace_back(index);
  }

  // initialize index2momentum table
  this->m_index2momentum_table.resize(this->m_num_k_stars, this->m_space_dim);
  int count = 0;
  // index of grids along z axis
  for (auto k = std::ceil(this->m_side_length / 2.0); k <= this->m_side_length;
       ++k) {
    // index of grids along x axis
    for (auto i = k; i <= this->m_side_length; ++i) {
      // index of grids along y axis
      for (auto j = k; j <= i; ++j) {
        this->m_index2momentum_table.row(count) =
            Eigen::Vector3d((double)i / this->m_side_length * 2 * M_PI - M_PI,
                            (double)j / this->m_side_length * 2 * M_PI - M_PI,
                            (double)k / this->m_side_length * 2 * M_PI - M_PI);
        count++;
      }
    }
  }
}

void Cubic::initial_nearest_neighbour_table() {
  // the coordination number for 3d cubic lattice is 6
  // correspondense between the table index and the direction of displacement :
  // 0: (x+1, y, z)    1: (x, y+1, z)    2: (x, y, z+1)
  // 3: (x-1, y, z)    4: (x, y-1, z)    5: (x, y, z-1)
  this->m_nearest_neighbour_table.resize(this->m_space_size,
                                         this->m_coordination_number);
  int L = this->m_side_length;
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      for (int k = 0; k < L; ++k) {
        int site_index = this->m_site_indexer.to_orbital({i, j, k});

        // Direction 0: (x+1, y, z)
        this->m_nearest_neighbour_table(site_index, 0) =
            this->m_site_indexer.to_orbital({(i + 1) % L, j, k});

        // Direction 1: (x, y+1, z)
        this->m_nearest_neighbour_table(site_index, 1) =
            this->m_site_indexer.to_orbital({i, (j + 1) % L, k});

        // Direction 2: (x, y, z+1)
        this->m_nearest_neighbour_table(site_index, 2) =
            this->m_site_indexer.to_orbital({i, j, (k + 1) % L});

        // Direction 3: (x-1, y, z)
        this->m_nearest_neighbour_table(site_index, 3) =
            this->m_site_indexer.to_orbital({(i - 1 + L) % L, j, k});

        // Direction 4: (x, y-1, z)
        this->m_nearest_neighbour_table(site_index, 4) =
            this->m_site_indexer.to_orbital({i, (j - 1 + L) % L, k});

        // Direction 5: (x, y, z-1)
        this->m_nearest_neighbour_table(site_index, 5) =
            this->m_site_indexer.to_orbital({i, j, (k - 1 + L) % L});
      }
    }
  }
}

void Cubic::initial_displacement_table() {
  this->m_displacement_table.resize(this->m_space_size, this->m_space_size);
  int L = this->m_side_length;
  for (auto i = 0; i < this->m_space_size; ++i) {
    const auto coords_i = this->m_site_indexer.from_orbital(i);
    const auto xi = coords_i[0];
    const auto yi = coords_i[1];
    const auto zi = coords_i[2];

    for (auto j = 0; j < this->m_space_size; ++j) {
      const auto coords_j = this->m_site_indexer.from_orbital(j);
      const auto xj = coords_j[0];
      const auto yj = coords_j[1];
      const auto zj = coords_j[2];

      // displacement pointing from site i to site j
      const auto dx = (xj - xi + L) % L;
      const auto dy = (yj - yi + L) % L;
      const auto dz = (zj - zi + L) % L;
      this->m_displacement_table(i, j) =
          this->m_site_indexer.to_orbital({dx, dy, dz});
    }
  }
}

void Cubic::initial_symmetry_points() {
  // high symmetry points of 3d cubic lattice
  // Gamma point:  ( 0,  0,  0)
  // X point:      (pi,  0,  0)
  // M point:      (pi, pi,  0)
  // R point:      (pi, pi, pi)
  this->m_gamma_point_index = 0;
  this->m_m_point_index = (std::floor(this->m_side_length / 2.0) + 1) *
                              (std::floor(this->m_side_length / 2.0) + 2) / 2 -
                          1;
  this->m_x_point_index =
      this->m_m_point_index - std::floor(this->m_side_length / 2.0);
  this->m_r_point_index = this->m_num_k_stars - 1;

  // high symmetry lines of 3d cubic lattice
  // Delta line:   (0,0,0)   ->  (pi,0,0)
  // Z line:       (pi,0,0)  ->  (pi,pi,0)
  // Sigma line:   (0,0,0)   ->  (pi,pi,0)
  // Lambda line:  (0,0,0)   ->  (pi,pi,pi)
  // S line:       (pi,0,0)  ->  (pi,pi,pi)
  // T line:       (pi,pi,0) ->  (pi,pi,pi)
  for (auto i = 0; i < std::floor(this->m_side_length / 2.0) + 1; ++i) {
    this->m_delta_line_index.emplace_back(i * (i + 1) / 2);
    this->m_z_line_index.emplace_back(this->m_x_point_index + i);
    this->m_sigma_line_index.emplace_back(i * (i + 3) / 2);

    if (i == 0) {
      this->m_lambda_line_index.emplace_back(this->m_gamma_point_index);
      this->m_s_line_index.emplace_back(this->m_x_point_index);
      this->m_t_line_index.emplace_back(this->m_m_point_index);
    } else {
      this->m_lambda_line_index.emplace_back(
          this->m_lambda_line_index.back() +
          (std::floor(this->m_side_length / 2.0) + 2 - i) *
              (std::floor(this->m_side_length / 2.0) + 3 - i) / 2);
      this->m_s_line_index.emplace_back(
          this->m_s_line_index.back() +
          (std::floor(this->m_side_length / 2.0) + 1 - i) *
              (std::floor(this->m_side_length / 2.0) + 2 - i) / 2 +
          1);
      this->m_t_line_index.emplace_back(
          this->m_t_line_index.back() +
          (std::floor(this->m_side_length / 2.0) + 1 - i) *
              (std::floor(this->m_side_length / 2.0) + 2 - i) / 2);
    }
  }
}

void Cubic::initial_fourier_factor_table() {
  // Re( exp(-ikx) ) for lattice site x and momentum k
  this->m_fourier_factor_table.resize(this->m_space_size, this->m_num_k_stars);
  for (auto x_index = 0; x_index < this->m_space_size; ++x_index) {
    for (auto k_index = 0; k_index < this->m_num_k_stars; ++k_index) {
      // this defines the inner product of a site vector x and a momemtum vector
      // k
      const auto site_coords = this->m_site_indexer.from_orbital(x_index);
      this->m_fourier_factor_table(x_index, k_index) =
          cos((-site_coords[0] * this->m_index2momentum_table(k_index, 0) -
               site_coords[1] * this->m_index2momentum_table(k_index, 1) -
               site_coords[2] * this->m_index2momentum_table(k_index, 2)));
    }
  }
}

void Cubic::initial_hopping_matrix() {
  this->m_hopping_matrix.resize(this->m_space_size, this->m_space_size);
  for (auto index = 0; index < this->m_space_size; ++index) {
    // direction 0 for x+1, 1 for y+1 and 2 for z+1
    const int index_xplus1 = this->NearestNeighbour(index, 0);
    const int index_yplus1 = this->NearestNeighbour(index, 1);
    const int index_zplus1 = this->NearestNeighbour(index, 2);

    this->m_hopping_matrix(index, index_xplus1) += 1.0;
    this->m_hopping_matrix(index_xplus1, index) += 1.0;
    this->m_hopping_matrix(index, index_yplus1) += 1.0;
    this->m_hopping_matrix(index_yplus1, index) += 1.0;
    this->m_hopping_matrix(index, index_zplus1) += 1.0;
    this->m_hopping_matrix(index_zplus1, index) += 1.0;
  }
}

void Cubic::initial() {
  // avoid multiple initialization
  if (!this->m_initial_status) {
    this->initial_site_indexer();
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
