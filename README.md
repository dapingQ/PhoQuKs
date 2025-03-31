# PhoQuKs

The data and code for Photonic Quantum KernelS.

## Installation 

The simulation is performed by [discopy](https://github.com/discopy/discopy).
Note the repo was under development and all results were simulated at the branch `optics` and the commit `a0aeb6c0471440250b06807802e7e94c748b4035`.

## Files

necessary packages and functions
- geo_diff.py

Data Folder
- exp_data, experiment data
- all_kernels_simulation, simulation data

Simualtion all Kernels
- simualte_all_kernels.py
- simulate_ntk.py

Simulate Noise and Error
- simulate_bs_ratio.py
- simulate_phase.py
- simulate_photon_statistics.py

All figure plots
- plot_main.py
- plot_SM.py

Other .npy file are used in plot scripts.
- transition_kernel.npy