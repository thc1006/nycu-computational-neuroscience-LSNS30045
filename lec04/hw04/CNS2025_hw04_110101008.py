import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def compute_sta(input_signal, spikes, dt_seconds, lag_times, demean=True):

    signal_length = input_signal.size

    # Calculate baseline mean if we're demeaning the result
    baseline = input_signal.mean() if demean else 0.0

    # Find all time points where spikes occurred
    spike_indices = np.where(spikes > 0)[0]

    # For each requested lag, compute the average input at that time relative to spikes
    sta_values = []
    for lag in lag_times:
        # Convert the lag time to an index offset
        time_shift = int(np.round(lag / dt_seconds))

        # Calculate where to look in the input signal
        shifted_indices = spike_indices + time_shift

        # Make sure we don't go out of bounds
        valid_mask = (shifted_indices >= 0) & (shifted_indices < signal_length)

        # Calculate the mean at this lag (or NaN if no valid data points)
        if np.any(valid_mask):
            lag_average = input_signal[shifted_indices[valid_mask]].mean() - baseline
        else:
            lag_average = np.nan

        sta_values.append(lag_average)

    return np.array(sta_values)


def softplus(x):
    return np.log1p(np.exp(x))


def lnp_kernel(dt, tau1=0.032, tau2=0.016, tau3=0.008, tmax=0.3):

    # Create time vector
    num_steps = int(np.round(tmax / dt)) + 1
    t = np.arange(0, num_steps) * dt

    # Combine three exponentials to get a biphasic shape
    k = np.exp(-t/tau1) - 1.8*np.exp(-t/tau2) + 0.8*np.exp(-t/tau3)

    # Force the first point to zero (no instantaneous effect)
    k[0] = 0.0

    # Normalize to peak at 1.0
    max_val = np.max(k)
    if max_val != 0:
        k = k / max_val

    return t, k


def gen_input_currents(num_samples, dt, rng, correlated=True):

    # Start with Gaussian white noise
    noise = rng.normal(size=num_samples)

    if not correlated:
        return noise

    # Apply a smoothing filter to introduce temporal correlations
    # This mimics more realistic, slowly-varying inputs
    filter_duration = 0.05  # 50 ms
    filter_tau = 0.01       # 10 ms time constant

    num_filter_points = int(np.round(filter_duration / dt)) + 1
    t_filter = np.arange(0, num_filter_points) * dt

    # Exponential smoothing kernel
    smoothing_filter = np.exp(-t_filter / filter_tau)
    smoothing_filter = smoothing_filter / smoothing_filter.sum()  # normalize

    # Convolve to get correlated input
    return np.convolve(noise, smoothing_filter, mode='same')


def lnp_spiketrain(input_current, kernel, dt, rng, r_target=10.0):

    # Step 1: Linear filtering
    linear_response = np.convolve(input_current, kernel, mode='same')

    # Step 2: Nonlinearity (normalize first, then apply softplus)
    normalized = (linear_response - linear_response.mean()) / (linear_response.std() + 1e-8)
    firing_rate = softplus(normalized)

    # Step 3: Scale to achieve target firing rate
    current_mean_rate = firing_rate.mean() / dt
    scaling_factor = r_target / current_mean_rate
    firing_rate = firing_rate * scaling_factor

    # Step 4: Generate Poisson spikes
    # For small dt, Poisson process approximates as Bernoulli trials
    spike_probability = firing_rate * dt
    spikes = (rng.uniform(size=firing_rate.shape) < spike_probability).astype(int)

    return spikes, firing_rate, linear_response


# ============================================================================
# Main analysis script
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Exercise 1: Basic STA computation on provided data
    # ------------------------------------------------------------------------
    print("Running Exercise 1...")

    # Load the data
    data1 = np.load("hw04-data.npz")
    input_current = data1['i'].astype(float).ravel()
    spike_train = data1['s'].astype(int).ravel()
    dt_milliseconds = float(data1['dt'])
    dt_sec = dt_milliseconds / 1000.0  # convert to seconds

    # Compute STA at several lags around the spike time
    lags_ms = np.array([-3, -2, -1, 0, 1, 2], dtype=float)
    lags_sec = lags_ms / 1000.0
    sta_result = compute_sta(input_current, spike_train, dt_sec, lags_sec, demean=True)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(lags_ms, sta_result, marker='o', linewidth=2, markersize=8)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.7, label='Spike time')
    plt.xlabel("Time lag (ms)", fontsize=12)
    plt.ylabel("STA (demeaned)", fontsize=12)
    plt.title("Exercise 1: Spike-Triggered Average", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------
    # Exercise 2: Blowfly H1 neuron data
    # ------------------------------------------------------------------------
    print("Running Exercise 2...")

    # Load Blowfly H1 data from MATLAB file
    blowfly_data = loadmat("c1p8.mat")
    stimulus = blowfly_data['stim'].astype(float).ravel()
    response = blowfly_data['rho'].astype(float).ravel()

    # Convert response to binary spike train (threshold at zero)
    spikes_blowfly = (response > 0).astype(int)

    # Time step is 2 ms (500 Hz sampling)
    dt_blowfly = 1.0 / 500.0

    # Compute STA over a wider range: -300 ms to +100 ms
    lags_blowfly_ms = np.arange(-300, 100 + 1, 2, dtype=float)
    lags_blowfly_sec = lags_blowfly_ms / 1000.0
    sta_blowfly = compute_sta(stimulus, spikes_blowfly, dt_blowfly,
                              lags_blowfly_sec, demean=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(lags_blowfly_ms, sta_blowfly, linewidth=1.5)
    plt.axvline(0.0, color='red', linestyle='--', alpha=0.7, label='Spike time')
    plt.xlabel("Time lag (ms)", fontsize=12)
    plt.ylabel("STA (demeaned)", fontsize=12)
    plt.title("Exercise 2: Blowfly H1 Neuron STA (-300 to +100 ms)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------
    # Exercise 3: LNP model simulations
    # ------------------------------------------------------------------------
    print("Running Exercise 3...")

    # Setup
    rng = np.random.default_rng(123)  # for reproducibility
    dt = 0.001  # 1 ms time step
    num_samples = 10_000

    # Generate the "true" kernel we'll try to recover
    kernel_time, true_kernel = lnp_kernel(dt)

    # Part A: Small dataset with correlated input
    print("  Part A: Small dataset (N=10k) with correlated input...")
    correlated_input = gen_input_currents(num_samples, dt, rng, correlated=True)
    spikes_small, rate_small, linear_small = lnp_spiketrain(
        correlated_input, true_kernel, dt, rng, r_target=10.0
    )

    # Compute STA
    lags_for_sta = np.arange(-0.3, 0.2 + dt, dt)
    sta_small = compute_sta(correlated_input, spikes_small, dt, lags_for_sta, demean=True)

    # Plot STA vs true kernel
    plt.figure(figsize=(10, 5))
    plt.plot(lags_for_sta * 1e3, sta_small, label="Recovered STA", linewidth=2)
    plt.plot(kernel_time * 1e3, true_kernel, label="True Kernel",
             linewidth=2, linestyle='--', alpha=0.7)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Time lag (ms)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Exercise 3A: STA Recovery (N=10k, correlated input)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Part B: Large dataset (to see convergence)
    print("  Part B: Large dataset (N=1M) with correlated input...")
    # Note: For real analysis, could use N=1e8, but that's slow
    # Using 1M here for demonstration
    num_samples_large = 1_000_000

    large_input = gen_input_currents(num_samples_large, dt, rng, correlated=True)
    spikes_large, rate_large, linear_large = lnp_spiketrain(
        large_input, true_kernel, dt, rng, r_target=10.0
    )
    sta_large = compute_sta(large_input, spikes_large, dt, lags_for_sta, demean=True)

    plt.figure(figsize=(10, 5))
    plt.plot(lags_for_sta * 1e3, sta_large, label="STA (N=1M)", linewidth=2)
    plt.plot(kernel_time * 1e3, true_kernel, label="True Kernel",
             linewidth=2, linestyle='--', alpha=0.7)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Time lag (ms)", fontsize=12)
    plt.ylabel("STA (demeaned)", fontsize=12)
    plt.title("Exercise 3B: STA with Large Dataset (N=1M)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Part C: White (uncorrelated) input
    print("  Part C: White noise input (N=1M)...")
    white_input = gen_input_currents(num_samples_large, dt, rng, correlated=False)
    spikes_white, rate_white, linear_white = lnp_spiketrain(
        white_input, true_kernel, dt, rng, r_target=10.0
    )
    sta_white = compute_sta(white_input, spikes_white, dt, lags_for_sta, demean=True)

    plt.figure(figsize=(10, 5))
    plt.plot(lags_for_sta * 1e3, sta_white, label="STA (white noise)", linewidth=2)
    plt.plot(kernel_time * 1e3, true_kernel, label="True Kernel",
             linewidth=2, linestyle='--', alpha=0.7)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Time lag (ms)", fontsize=12)
    plt.ylabel("STA (demeaned)", fontsize=12)
    plt.title("Exercise 3C: STA with Uncorrelated (White) Input", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("All exercises complete!")
