# Documentation for `tbfit.f90`

## Overview

`tbfit.f90` is the main program of the TBFIT (Tight-Binding Parameter Fitting) package. It orchestrates the overall workflow of the tight-binding parameter fitting process and subsequent calculations. The program reads input parameters, performs the fitting, executes post-processing tasks, and generates output files, including plots if requested.

## Key Components

The program `tbfit` is structured as a sequence of calls to various subroutines and modules that handle specific parts of the calculation.

- **Initialization:**
    - `parse_very_init(PINPT)`: Parses initial command-line arguments or input file to get basic settings.
    - Allocates memory for various data structures based on the initial input.
    - MPI setup (`mpi_initialize`): Initializes the MPI environment if the program is compiled with MPI support.
    - Opens a log file.
- **Main Processing:**
    - `version_stamp(t_start)`: Records the start time and prints version information.
    - `parse(PINPT)`: Parses the main input file(s) to load all necessary parameters, system geometry, k-points, DFT energies, etc.
    - `test()`: Calls a testing routine if compiled with a test flag.
    - `get_fit(PINPT, PPRAM_FIT, PKPTS, EDFT, PWGHT, PGEOM, NN_TABLE, PINPT_BERRY, PINPT_DOS)`: This is the core routine that performs the tight-binding parameter fitting. It takes various inputs (DFT energies, geometry, k-points) and outputs the fitted parameters (`PPRAM_FIT`).
    - `post_process(PINPT, PPRAM, PPRAM_FIT, PKPTS, EDFT, PWGHT, PGEOM, NN_TABLE, PINPT_BERRY, PINPT_DOS, PRPLT)`: Performs calculations after the fitting, such as band structure calculation with the fitted parameters, density of states, Berry phase related quantities, etc.
    - `get_replot(PINPT, PGEOM, PKPTS, PRPLT)`: Generates data for replotting purposes.
- **Finalization:**
    - Prints program end timestamp and total elapsed time.
    - `execute_command_line(gnu_command)`: If plotting is enabled (`PINPT%flag_plot`), it executes Gnuplot to generate plots.
    - MPI finalization (`mpi_finish`): Finalizes the MPI environment if used.
    - Closes the log file.

## Important Variables/Constants

The program uses several derived types to manage data. These are typically defined in the `parameters` module.

- `PINPT` (type `incar`): Holds input parameters read from the main input file. Controls the behavior of the program.
- `PPRAM_FIT` (type `params`): Stores the tight-binding parameters obtained from the fitting process.
- `PPRAM` (type `params`, allocatable array): Stores input tight-binding parameters for different systems.
- `PKPTS` (type `kpoints`, allocatable array): Stores k-point information.
- `EDFT` (type `energy`, allocatable array): Stores reference energies (e.g., from DFT calculations).
- `ETBA` (type `energy`, allocatable array): Stores energies calculated by the tight-binding model.
- `PWGHT` (type `weight`, allocatable array): Stores weights for fitting.
- `PGEOM` (type `poscar`, allocatable array): Stores atomic geometry information.
- `NN_TABLE` (type `hopping`, allocatable array): Stores nearest-neighbor hopping information.
- `PINPT_BERRY` (type `berry`, allocatable array): Stores input parameters for Berry phase calculations.
- `PINPT_DOS` (type `dos`, allocatable array): Stores input parameters for Density of States calculations.
- `PRPLT` (type `replot`, allocatable array): Stores data and parameters for replotting.

## Usage Examples

The program is typically compiled using a makefile. The `README.md` and `PYTHON_MODULE/README` provide general instructions on compilation and execution.

To run the program:
```bash
# For serial execution (example)
./tbfit < input_file

# For MPI execution (example, actual command may vary)
mpirun -np <num_processes> ./tbfit.mpi < input_file
```
The specific input file format and command-line arguments are crucial and would be detailed in the user manual (`MANUAL/tbfit_manual.pdf`).

## Dependencies and Interactions

- **Modules:**
    - `parameters`: Defines derived data types and constants.
    - `mpi_setup`: Handles MPI initialization and finalization.
    - `time`: Provides timing utilities.
    - `version`: Provides version information.
    - `print_io`: Handles printing messages and logging.
    - Other modules likely involved in `get_fit` and `post_process` (e.g., for specific algorithms or calculations).
- **External Subroutines:**
    - `get_eig`: An external subroutine, likely for eigenvalue calculations.
- **Input Files:**
    - A main input file (format specific to TBFIT) providing control parameters, system details, etc.
    - Files containing DFT band structures, k-points, atomic positions.
- **Output Files:**
    - Log file.
    - Output files with fitted parameters.
    - Data files for band structures, DOS, Berry curvature, etc.
    - Plot files (if Gnuplot is used).
- **Libraries:**
    - MPI library (if compiled for parallel execution).
    - Potentially other numerical libraries linked during compilation (e.g., LAPACK, ScaLAPACK).
- **Includes:**
    - `alias.inc`: This file likely contains preprocessor directives or common aliases used throughout the Fortran source code.

This program acts as the central driver, coordinating the reading of inputs, execution of core computational routines, and generation of outputs.
