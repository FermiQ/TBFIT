# Documentation for `tbfitpy_serial.py`

## Overview

`tbfitpy_serial.py` is a Python script that provides a high-level interface to the TBFIT (Tight-Binding Parameter Fitting) package for serial (non-MPI) execution. It allows users to initialize and control tight-binding calculations, perform parameter fitting, analyze results, and visualize data such as band structures. The script makes extensive use of NumPy for numerical operations and Matplotlib for plotting. It also interfaces with a compiled Fortran backend (`tbfitpy_mod.pyfit`) for computationally intensive tasks.

The script defines several classes: `mycolor` and `myfont` for plot styling, `pytbfit` as the main class for TBFIT operations, and `csa_tools` and `csa_soldier_tools` which appear to be related to a Clonal Selection Algorithm (CSA) for optimization.

## Key Components

### 1. `mycolor` Class
- **Purpose:** Defines dictionaries for consistent color and marker styling in plots, based on orbital types.
- **Key Attributes:**
    - `orb_index`: Maps orbital names (e.g., 's', 'px', 'dxy') to numerical indices.
    - `marker_index`: Maps orbital names to Matplotlib marker styles.
    - `color_index`: Maps orbital names to color pairs (likely for different spin states or representations).

### 2. `myfont` Class
- **Purpose:** Defines a default font dictionary for Matplotlib plots.
- **Key Attributes:**
    - `font`: A dictionary specifying font family, color, weight, and size.

### 3. `pytbfit` Class
- **Purpose:** The main class for interacting with the TBFIT Fortran backend and managing calculations.
- **Initialization (`__init__`)**:
    - Parses command-line arguments (`-p` for parameter file, `-red_ovl` for overlap reduction, `-red_hop` for hopping reduction).
    - Handles MPI communicator (though this is the serial version, it has placeholder for MPI).
    - Initializes Fortran data structures by calling `pyfit.init_incar_py`, `pyfit.init_params_py`, etc. These `pyfit` functions are from the imported `tbfitpy_mod` module, which is the Python interface to the compiled Fortran library.
    - Supports handling multiple systems/input files (`nsystem`).
- **Core Methods:**
    - `init(verbose, orbfit, myid, pfilenm, red_ovl, red_hop)`: Initializes the calculation by calling the Fortran `init` routines (`pyfit.init`, `pyfit.init2`, etc., depending on `nsystem`). Loads target band structure data and orbital indexing.
    - `orb_index(pgeom)`: Creates dictionaries mapping orbital names (per species and per atom) to their numerical indices in the basis set.
    - `load_band(pwght, pgeom)`: Loads target band structure energies and orbital projections from a file.
    - `constraint_param(iparam_type, iparam, fix, ub, lb)`: Sets constraints on parameters (fix values, set upper/lower bounds).
    - `set_param_bound(bnds, param_idx)`: Helper to set parameter bounds.
    - `set_param_fix(fix, param_idx)`: Helper to fix/unfix parameters.
    - `fit(verbose, miter, tol, pso_miter, method, pso_options, n_particles, iseed, sigma, sigma_orb)`: Performs parameter fitting using specified methods ('lmdif', 'mypso', 'mypso.lmdif'). Calls corresponding Fortran routines (`pyfit.fit`, `pyfit.pso`). Calculates cost functions.
    - `get_eig(verbose, sys)`: Calculates eigenvalues (band structure) using the current parameters. Calls Fortran `pyfit.eig`.
    - `generate_TBdata(filenm, ndata, N_fit, tol, method, myid)`: Appears to generate a dataset of tight-binding calculations, possibly for machine learning.
    - `copy_param_best(imode)`: Copies parameters (e.g., best parameters to current or vice-versa) by calling `pyfit.copy_params_best`.
    - `toten(eltemp, nelect)`: Calculates total energy by calling `pyfit.toten`.
    - `get_kpath(pkpts)`: Extracts k-path information (distances, special k-point names and positions) for plotting.
    - `plot_line(...)`, `plot_dots(...)`: Helper methods for Matplotlib band structure plotting (lines and scatter points for orbital projections).
    - `get_proj_weight(v2, i_atoms, i_specs, proj)`: Calculates projected weights for orbitals/atoms/species.
    - `get_system(isystem)`: Retrieves data (energies, k-path, title) for a specific system if multiple are loaded.
    - `get_proj_ldos(isystem)`: Retrieves projected LDOS data.
    - `set_proj_c(projs)`, `set_proj_m(projs)`: Sets colors and markers for projections based on `mycolor` class.
    - `get_proj_index(isystem)`: Gets orbital projection indices for a system.
    - `get_grid_ratio(nsystem, sys)`: Calculates ratios of k-path lengths for subplotting multiple systems.
    - `plot_band(...)`: The main band structure plotting function. It can plot target DFT bands, TBA fitted bands, and orbital projections.
    - `plot_pso_cost_history(...)`: Plots the cost function history during PSO.
    - `plot_pso_pbest(...)`: Plots the best parameters found by particles in PSO.
    - `print_param(param_out)`: Saves fitted parameters to a file by calling `pyfit.print_param_py`.
    - `print_weight(weight_out)`: Saves fitting weights to a file.
    - `load_weight(weight_in)`: Loads fitting weights from a file.
    - `select_param(param_set, i)`: Selects a parameter value considering constraints (e.g., if fixed or tied to another).
    - `print_fit(suffix)`: Prints/saves target and fitted band structure data files.
    - `save(...)`: A comprehensive method to save various outputs: plots, data files (target bands, fitted bands, parameters, weights, cost histories).
    - `plot_param(...)`: Plots parameter sets, possibly from a file of best particles in an optimization.

### 4. `csa_tools` Class
- **Purpose:** Implements tools for a Clonal Selection Algorithm (CSA), a type of evolutionary algorithm for optimization. This class seems to manage the overall CSA process.
- **Key Methods:**
    - `init()`: Initializes CSA parameters, loads bounds for variables.
    - `repulsion()`, `coulomb()`: Likely related to maintaining diversity in the population of solutions.
    - `gen_directories()`, `del_directories()`: Manages directories for parallel CSA "soldiers" (individual evaluations).
    - `load_solutions()`, `save_solutions()`: Load/save population state from/to a dump file.
    - `gen_trial_solution()`, `write_trial_solution()`: Generates and writes input files for soldier processes.
    - `get_solution()`: Retrieves results from completed soldier processes.
    - `mutation()`, `crossover()`: Genetic operators for CSA.
    - `selection()`, `gen_parents()`: Selects parent solutions for generating new candidates.
    - `csa_initial_step()`, `csa_main_step()`, `csa_run()`: Orchestrate the CSA optimization lifecycle.

### 5. `csa_soldier_tools` Class
- **Purpose:** Tools for the "soldier" processes in the CSA framework. Each soldier likely evaluates a single candidate solution.
- **Key Methods:**
    - `__init__(param_type, param_const)`: Initializes based on parameter types and constraints from `pytbfit`. Determines the number of parameters to be optimized (`self.nparam`).
    - `read_csa_input()`: Reads a candidate solution's parameters from an input file.
    - `write_csa_output()`: Writes the evaluation result (objective function value) to an output file.
    - `check_obj_rank()`: Checks if a new solution is better than existing ones in a dump file.
    - `update_csa_result()`: Updates result files if a better solution is found.
    - `update_status()`: Updates status files to signal completion.

## Important Variables/Constants

- **Module-level:**
    - `pyfit`: This is the imported module from `tbfitpy_mod`, which serves as the bridge to the compiled Fortran library. All low-level computations (diagonalization, fitting algorithms) are called through `pyfit`.
- **`pytbfit` Class Instance Variables:**
    - `self.pinpt`, `self.ppram`, `self.pkpts`, `self.pwght`, `self.pgeom`, `self.hopping`, `self.edft`, `self.etba`: These are Python representations of Fortran derived types, holding all the data for the tight-binding model, parameters, k-points, energies, weights, geometry, etc. They are initialized and populated by calls to `pyfit` functions.
    - `self.nsystem`: Number of systems being handled.
    - `self.cost_history`, `self.cost`, `self.cost_orb`, `self.cost_total`: Store cost function values during fitting.

## Usage Examples

The script is intended to be used as a library or a main script for performing tight-binding calculations and fitting.

```python
# Example: Initialize and perform a fit
import tbfitpy_serial as tbfit

# Create an instance for a single input file 'INCAR-TB'
fitter = tbfit.pytbfit(filenm='INCAR-TB')

# Initialize the system, load DFT bands, etc.
# pfilenm specifies the parameter file to use/generate.
fitter.init(verbose=True, pfilenm='PARAM_FIT.dat')

# Optionally, set parameter constraints
# fitter.constraint_param(iparam_type=1, fix=True) # Fix all onsite energies
# fitter.constraint_param(iparam=0, lb=-10.0, ub=-5.0) # Bounds for first parameter

# Perform the fit using Levenberg-Marquardt
fitter.fit(method='lmdif', miter=100, tol=1e-6)

# Get eigenvalues with fitted parameters
fitter.get_eig()

# Plot the band structure
fitter.plot_band(fout='band_fitted.pdf', title='My System',
                 plot_target=True, yin=-5, yen=5)

# Save the fitted parameters
fitter.print_param(param_out='PARAM_FIT.new.dat')

# Save various outputs
fitter.save(title='lmdif_run', plot_band=True, param=True, band=True, target=True, cost_history=True)
```

For CSA optimization:
```python
# (Assuming CSA_SOLDIER.py is set up and INCAR-TB/bnds.txt are configured for CSA)
# This part is more speculative as it requires external setup.

# Info_csa might be a class or SimpleNamespace
# Info_csa.npop = 20  # Number of population
# Info_csa.ndim = fitter.ppram.nparam_free # Number of free parameters to optimize
# Info_csa.apath = '.' # Working directory
# Info_csa.fname_dump = 'csa_dump.txt'

# csa_manager = tbfit.csa_tools(Info_csa)
# csa_manager.init()
# csa_manager.gen_directories() # If soldiers run in separate dirs
# csa_manager.csa_run()
```

## Dependencies and Interactions

- **External Libraries:**
    - `numpy`: For numerical arrays and operations.
    - `matplotlib.pyplot`, `matplotlib.mpl`: For plotting.
    - `tqdm`: For progress bars.
    - `torch`: Used in `generate_TBdata`, suggesting an interface to PyTorch for machine learning applications.
    - `sys`, `os`, `gc`, `time`, `random`, `warnings`: Standard Python libraries.
- **TBFIT Fortran Backend:**
    - `tbfitpy_mod.pyfit`: This is the crucial dependency. It's the Python wrapper (likely generated by f2py or similar) around the compiled Fortran routines that perform the core calculations.
- **Input Files:**
    - `INCAR-TB` (or user-specified): Main input file for TBFIT settings.
    - Parameter files (e.g., `PARAM_FIT.dat`): Contain tight-binding parameters.
    - Target band structure files (name usually specified in `INCAR-TB`).
    - `bnds.txt` (for CSA): Specifies bounds for parameters during CSA optimization.
- **Output Files:**
    - Plot files (e.g., `band.pdf`, `COST_HISTORY.pdf`).
    - Data files (`PARAM_FIT.new.dat`, `band_structure_DFT.dat`, `band_structure_TBA.dat`, `WEIGHT.dat`, etc.).
    - Log files or console output (controlled by `verbose` flags).
    - CSA related files (`dump.txt`, files in soldier directories).

The `tbfitpy_serial.py` script acts as a comprehensive Python suite for leveraging the TBFIT Fortran engine, providing high-level commands for complex workflows like parameter fitting, band structure calculation, and advanced optimization using CSA.
