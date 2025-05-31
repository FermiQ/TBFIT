# Documentation for `tbfitpy_mpi.py`

## Overview

`tbfitpy_mpi.py` is a Python script that provides a high-level interface to the TBFIT (Tight-Binding Parameter Fitting) package, specifically tailored for MPI (Message Passing Interface) parallel execution. It allows users to initialize and control tight-binding calculations, perform parameter fitting, analyze results, and visualize data such as band structures, leveraging multiple processors for enhanced performance.

This script is largely similar in functionality and structure to `tbfitpy_serial.py`. The primary distinctions are its use of `mpi4py` for MPI communication and its interface with an MPI-enabled Fortran backend (`tbfitpy_mod_mpi.pyfit`).

The script defines several classes: `mycolor` and `myfont` for plot styling, `pytbfit` as the main class for TBFIT operations, and `csa_tools` and `csa_soldier_tools` which appear to be related to a Clonal Selection Algorithm (CSA) for optimization (these CSA classes seem to operate similarly to their serial counterparts but might leverage the MPI environment if their "soldier" processes are MPI-aware).

## Key Components

The classes and their methods are largely identical to those in `tbfitpy_serial.py`. Below are the key distinctions related to MPI:

### 1. `pytbfit` Class
- **Purpose:** The main class for interacting with the TBFIT Fortran backend and managing calculations in an MPI environment.
- **Initialization (`__init__`)**:
    - **MPI Handling**:
        - Takes an optional `mpicomm` argument (an `mpi4py` communicator).
        - If `mpicomm` is provided, it stores it as `self.comm` and converts it to a Fortran-compatible MPI communicator using `self.fcomm = self.comm.py2f()`. This `self.fcomm` is then passed to the underlying Fortran routines.
        - If `mpicomm` is `None` (e.g., running in a pseudo-serial mode or by a single MPI process), `self.fcomm` is set to 0, similar to the serial version.
    - Parses command-line arguments (`-p`, `-red_ovl`, `-red_hop`) as in the serial version.
    - Initializes Fortran data structures by calling `pyfit.init_incar_py`, `pyfit.init_params_py`, etc. These `pyfit` functions are from the imported `tbfitpy_mod_mpi` module, which is the Python interface to the MPI-enabled compiled Fortran library.
    - Supports handling multiple systems/input files (`nsystem`).
- **Core Methods:**
    - Most methods (e.g., `init`, `fit`, `get_eig`, plotting functions, saving functions) are identical in signature and high-level logic to the serial version. The key difference is that they pass `self.fcomm` (the Fortran MPI communicator) to the underlying `pyfit` functions from `tbfitpy_mod_mpi`. This allows the Fortran backend to distribute computations across MPI processes.
    - Operations that are purely Python-based or client-side (e.g., most of the plotting logic, parameter setup before calling Fortran) remain the same.
    - The `myid` parameter in methods like `init` and `save` is used to ensure that certain operations (like file I/O or printing to console) are performed only by a specific MPI rank (typically rank 0) to avoid race conditions or redundant output.

### Other Classes (`mycolor`, `myfont`, `csa_tools`, `csa_soldier_tools`)
- These classes and their methods appear to be identical to their counterparts in `tbfitpy_serial.py`.
- The `csa_tools` and `csa_soldier_tools` might implicitly benefit from or require an MPI environment if the "soldier" tasks they manage are themselves MPI programs or are launched in a way that utilizes the parallel environment set up by `tbfitpy_mpi.py`. The provided `csa_soldier_command` in `csa_tools` might be an `srun` or `mpirun` command.

## Important Variables/Constants

- **Module-level:**
    - `MPI`: The imported `MPI` module from `mpi4py`.
    - `pyfit`: This is the imported module from `tbfitpy_mod_mpi`, serving as the bridge to the MPI-enabled compiled Fortran library.
- **`pytbfit` Class Instance Variables:**
    - `self.comm`: The `mpi4py` communicator.
    - `self.fcomm`: The Fortran MPI communicator derived from `self.comm`.
    - Other instance variables (`self.pinpt`, `self.ppram`, etc.) are the same as in the serial version.

## Usage Examples

The script is intended to be run in an MPI environment, for example, using `mpirun` or `srun`.

```python
# Example: Initialize and perform a fit in an MPI script
from mpi4py import MPI
import tbfitpy_mpi as tbfit # Assuming the script is named tbfitpy_mpi.py

# Get the default MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Get rank of the current process

# Create an instance for a single input file 'INCAR-TB', passing the communicator
fitter = tbfit.pytbfit(mpicomm=comm, filenm='INCAR-TB')

# Initialize the system (all MPI processes will call this)
# pfilenm specifies the parameter file to use/generate.
# myid=rank ensures file operations within init might be rank-specific if needed by Fortran layer
fitter.init(verbose=(rank == 0), pfilenm='PARAM_FIT.dat', myid=rank)

# Perform the fit (collective operation)
fitter.fit(method='lmdif', miter=100, tol=1e-6, verbose=(rank == 0))

# Get eigenvalues (collective operation)
fitter.get_eig(verbose=(rank == 0))

# Plotting and saving usually done by rank 0 to avoid multiple files/outputs
if rank == 0:
    fitter.plot_band(fout='band_fitted_mpi.pdf', title='My System MPI',
                     plot_target=True, yin=-5, yen=5)
    fitter.print_param(param_out='PARAM_FIT.new_mpi.dat')
    fitter.save(title='lmdif_run_mpi', plot_band=True, param=True, band=True, target=True,
                  cost_history=True, myid=rank) # Pass myid to save method
```

The `PYTHON_MODULE/README` provides command-line examples:
```bash
mpirun -np 32 pytbfit # pytbfit here would refer to tbfitpy_mpi.py
# or
srun --mpi=pmi2 -ntasks=32 pytbfit
```

## Dependencies and Interactions

- **External Libraries:**
    - `mpi4py`: Essential for MPI communication.
    - `numpy`, `matplotlib.pyplot`, `matplotlib.mpl`, `tqdm`, `torch`, `sys`, `os`, `gc`, `time`, `random`, `warnings`: Same as the serial version.
- **TBFIT Fortran Backend:**
    - `tbfitpy_mod_mpi.pyfit`: The Python wrapper around the MPI-enabled compiled Fortran library.
- **Input/Output Files:**
    - Same as the serial version. File I/O should typically be managed by a single process (e.g., rank 0) to prevent conflicts in a parallel file system, unless the underlying Fortran library handles parallel I/O. The `myid` argument in `save()` and `init()` methods is used for this purpose.

`tbfitpy_mpi.py` extends the capabilities of `tbfitpy_serial.py` by enabling parallel execution through MPI. This is crucial for handling larger systems or more computationally demanding fitting tasks where the Fortran backend can distribute work across multiple processors. The Python script itself orchestrates these parallel operations by passing the MPI communicator to the Fortran layer.
