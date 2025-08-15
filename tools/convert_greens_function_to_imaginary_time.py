import numpy as np
import os

def parse_greens_function_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Green's function file not found: {filepath}")

    G_k_iw = None

    with open(filepath, 'r') as f:
        header_line = f.readline().strip().split()
        if len(header_line) < 2:
            raise ValueError(f"Invalid header in {filepath}. Expected 'num_channels num_frequencies'.")

        try:
            num_channels = int(header_line[0])
            num_frequencies = int(header_line[1])
        except ValueError:
            raise ValueError(f"Could not parse num_channels or num_frequencies from header in {filepath}.")

        G_k_iw = np.zeros((num_channels, num_frequencies))

        for line_num, line in enumerate(f):
            parts = line.strip().split()

            try:
                channel_idx = int(parts[0])
                freq_idx = int(parts[1])
                value = float(parts[2])
            except ValueError:
                print(f"Warning: Could not parse channel or frequency index on line {line_num} in {filepath}. Skipping.")
                continue

            G_k_iw[channel_idx][freq_idx] = value

        return num_channels, num_frequencies, G_k_iw

def parse_imaginary_time_grid_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Imaginary time grid file not found: {filepath}")

    tau_grid = None

    with open(filepath, 'r') as f:
        header_line = f.readline().strip().split()
        if len(header_line) < 3:
            raise ValueError(f"Invalid header in {filepath}. Expected 'num_points beta delta_tau'.")

        try:
            total_tau_points = int(header_line[0])
            beta = float(header_line[1])
            delta_tau = float(header_line[2])
        except ValueError:
            raise ValueError(f"Could not parse header values from {filepath}.")

        tau_grid = np.zeros(total_tau_points)

        for line_num, line in enumerate(f):
            parts = line.strip().split()
            try:
                tau_idx = int(parts[0])
                tau_value = float(parts[1])
            except ValueError:
                print(f"Warning: Could not parse tau value on line {line_num} in {filepath}. Skipping.")
            tau_grid[tau_idx] = tau_value

    return beta, tau_grid


def calculate_matsubara_frequencies(beta, num_frequencies):
    n_indices = np.arange(num_frequencies)
    omega_n = (2 * n_indices + 1) * np.pi / beta
    return omega_n

def fourier_transform_matsubara(beta, G_iw_n, omega_n, tau_grid):
    """
    Performs the inverse Fourier transform from Matsubara frequencies to imaginary time.

    G(tau) = (1/beta) * sum_n [ G(i omega_n) * exp(-i * omega_n * tau) ]
    """
    cos_matrix = np.cos(np.outer(omega_n, tau_grid))
    G_tau = (2 / beta) * (G_iw_n @ cos_matrix)

    return G_tau

def main():
    greens_function_filepath = 'greens_functions.txt'
    num_channels, num_frequencies, G_k_iw = parse_greens_function_data(greens_function_filepath)
    print(num_frequencies, num_frequencies, G_k_iw)

    imaginary_tau_grid_filepath = 'imaginary_time_grids.txt'
    beta, tau_grid = parse_imaginary_time_grid_data(imaginary_tau_grid_filepath)
    print(beta, tau_grid)

    omega_n = calculate_matsubara_frequencies(beta, num_frequencies)
    print(omega_n)

    G_tau = fourier_transform_matsubara(beta, G_k_iw[0], omega_n, tau_grid)
    print(G_tau)

    for i in range(len(tau_grid)):
        print(tau_grid[i], np.real(G_tau[i]), np.imag(G_tau[i]))

if __name__ == "__main__":
    main()
