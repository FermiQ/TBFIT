# $^ : dependency list
# $@ : target

#----- Change options and library path according to your system ------------#
#-----------------------------------
# Compiler options and bin path    |
#---------------------------------------------------------------------------|
#OPTIONS= -fpp -DMPI -mcmodel=large # for curion2
################# Possible options ##########################################
#  -DSPGLIB      : printout spacegroup information in the initial stages
#                : if you want to use this option, please link SPGLIB 
#                  library path properly in the "Dependencies" section below
#  -DMPI         : MPI paralallism activation 
#                  If -DSCALAPACK option is activated: k-point + eigenvalue 
#                  parallism will be imployed,
#                  otherwise, only the k-point parallization will be performed.   
#  -DF08         : Fortran 2008 language is accepted or not
#  -DMKL_SPARSE  : use MKL_SPARSE library for the sparse matrix routines
#                  This option will save your memory 'enormously'.
#                  Before activate this option, make sure that the file
#                  mkl_spblas.f90 is exist in $(MKLPATH)/include/mkl_spblas.f90
#                  If this option is activated, you can use EWINDOW tag in
#                  your input file. See the manual for the details.
#  -DPSPARSE     : use FEAST_MPI with 4.0 version instead of MKL FEAST_SMP 2.0 version.
#                  This option is only available if -DMKL_SPARSE is activated
#                  For the details, please go to http://www.ecs.umass.edu/~polizzi/feast/
#                  NOTE: not a valid option in the current version. On developing now...
#  -DSCALAPACK   : use ScaLAPACK library for the eigenvalue parallism 
#                  !!! WARN !!! do not use in the current version: it is upon
#                               developing stage now.
#############################################################################
#FC   = mpif90 
 TBBIN= $(HOME)/code/bin
 VERSION=$(shell date +%Y%m%d)
#VERSION=0.41
 
# MAC-INTEL COMPILE
 OPTIONS= -fpp -DMPI -DF08 -DSPGLIB -DMKL_SPARSE -DPSPARSE #-DSCALAPACK 
 F90    = $(FC) $(OPTIONS)
 FFLAG  = -O2 -heap-arrays -nogen-interfaces
# LINUX-gfortran COMPILE
#OPTIONS= #-DMPI
#F90    = gfortran-mp-8 $(OPTIONS)
#FFLAG  = -cpp -O2 -ffree-line-length-256 -fmax-stack-var-size=32768

#OPTIONS= -cpp -DMPI -DF08 -DSPGLIB #-DMKL_SPARSE -DSCALAPACK
#F90    = mpif90-openmpi-mp $(OPTIONS)
#FFLAG  = -O2 -ffree-line-length-512 -fmax-stack-var-size=32768
#BIN    = ~/code/bin
BIN    = $(TBBIN)
#---------------------------------------------------------------------------|

#-----------------------------------
# Dependencies: LAPACK, SPGLIB     |
#---------------------------------------------------------------------------|
#SPGLIB    = -L/Users/Infant/code/lib/ -lsymspg   # home
SPGLIB    = -L/${HOME}/tbfit_fortran/LIB/spglib-master -lsymspg
#SPGLIB    = -L/home/Infant/tbfit_fortran/LIB/spglib-master -lsymspg  # curion2

MKLPATH   = $(MKLROOT)
LAPACK    = -L$(MKLPATH)/lib/ \
            -lmkl_intel_lp64 -lmkl_sequential \
            -lmkl_core -liomp5
BLAS      = 
INCLUDE   = -I$(MKLPATH)/include
#FEAST_MPI = -L/${HOME}/tbfit_fortran/LIB/FEAST/3.0/lib/x64  -lpfeast_sparse -lpfeast 
FEAST_MPI = -L/${HOME}/tbfit_fortran/LIB/FEAST/4.0/lib/x64  -lfeast  # MPI version need to be included in the future
#SCALAPACK = /Users/Infant/tbfit_fortran/LIB/scalapack-2.0.2/libscalapack.a
SCALAPACK = /${HOME}/tbfit_fortran/LIB/scala_home/libscalapack.a
#---------------------------------------------------------------------------|


######################### Do not modify below ###############################
#-----------------------------------
# Objects                          |
#---------------------------------------------------------------------------|
MKL_SP =$(findstring -DMKL_SPARSE,$(OPTIONS))
MKL_SPARSE = mkl_spblas.o
ifeq ($(MKL_SP),-DMKL_SPARSE)
  SP_MOD = mkl_spblas.o
else 
  SP_MOD = 
endif
SCALAPACK_USE=$(findstring -DSCALAPACK,$(OPTIONS))
SPARSE_PARA=$(findstring -DPSPARSE,$(OPTIONS))

MPI_MOD= blacs_basics.o mpi_basics.o mpi_setup.o 
TEST   = test.o
MODULE = print_io.o $(MPI_MOD) memory.o time.o version.o $(SP_MOD) \
		 parameters.o set_default.o  element_info.o read_incar.o \
		 orbital_wavefunction.o kronecker_prod.o phase_factor.o \
		 do_math.o print_matrix.o sorting.o berry_phase.o sparse_tool.o \
		 pikaia_module.o geodesiclm.o kill.o get_parameter.o \
		 reorder_band.o total_energy.o projected_band.o cost_function.o
READER = parse.o read_input.o read_param.o read_poscar.o read_kpoint.o \
		 read_energy.o set_weight.o get_site_number.o find_nn.o
WRITER = print_param.o plot_eigen_state.o plot_stm_image.o set_ribbon_geom.o print_energy.o \
		 print_wcc.o print_zak_phase.o print_berry_curvature.o replot_dos_band.o \
         print_circ_dichroism.o
GET    = get_tij.o get_eig.o get_dos.o get_soc.o get_param_class.o \
		 get_cc_param.o get_berry_curvature.o get_wcc.o get_zak_phase.o \
         get_z2_invariant.o get_parity.o get_symmetry_eig.o get_hamk_sparse.o \
         get_effective_ham.o e_onsite.o get_degeneracy.o get_circular_dichroism.o \
		 post_process.o
SYMM   = get_symmetry.o 
SPG_INT= spglib_interface.o
FITTING_LIB= get_fit.o minpack_sub.o lmdif.o genetic_alorithm.o

ifeq ($(SCALAPACK_USE), -DSCALAPACK)
  SCALAPACK_LIB= $(SCALAPACK)
  SCALAPACK_OBJ= #scalapack_initialize.o
else
  SCALAPACK_LIB=
  SCALAPACK_OBJ= 
endif

ifeq ($(SPARSE_PARA), -DPSPARSE)
  FEAST_LIB= $(FEAST_MPI)
else
  FEAST_LIB= 
endif

SPG    =$(findstring -DSPGLIB,$(OPTIONS))

ifeq ($(SPG),-DSPGLIB)
  SPGLIB_=  $(SPGLIB)
  OBJECTS=  $(MODULE) tbfit.o tbpack.o $(READER) $(WRITER) $(GET) \
                      $(FITTING_LIB) $(SCALAPACK_OBJ) $(TEST) $(SPG_INT) $(SYMM)
else
  SPGLIB_= 
  OBJECTS=  $(MODULE) tbfit.o tbpack.o $(READER) $(WRITER) $(GET) \
                      $(FITTING_LIB) $(SCALAPACK_OBJ) $(TEST) $(SYMM)
endif


#---------------------------------------------------------------------------|

#-----------------------------------
# Suffix rules                     |
#-----------------------------------
ifeq ($(MKL_SP),-DMKL_SPARSE)
.SUFFIXES: $(MKL_SPARSE)
$(MKL_SPARSE): $(MKLPATH)/include/mkl_spblas.f90
	$(F90) $(FFLAG) -c $<
endif
.SUFFIXES: .f .f90 
%.o: %.f90
	$(F90) $(FFLAG) -c $<

#$(F90) $(FFLAG) -Wl,-rpath,. -c $<


#-----------------------------------
# Targets                          |
#-----------------------------------
#$(BIN)/tbfit: $(OBJECTS) 

#tbfit.$(VERSION): $(OBJECTS) 
tbfit: $(OBJECTS) 
	$(F90) -o $@ $^ $(FEAST_LIB) $(BLAS) $(LAPACK) $(SCALAPACK_LIB) $(SPGLIB_) $(INCLUDE)
	cp tbfit $(BIN)/tbfit
	cp tbfit tbfit.$(VERSION)
	if [ -d "./tbfit.versions" ]; then cp tbfit.$(VERSION) tbfit.versions ; fi

get_ldos: print_io.o $(MPI_MOD) phase_factor.o do_math.o get_ldos.o 
	$(F90) -o $@ $^ $(LAPACK)
	cp get_ldos $(BIN)

#poscar2bs: poscar2bs.o
#	$(F90) -o $@ $^ 
#	cp poscar2bs $(BIN)/poscar2bs

fl2xyz: fleur2bs.o
	$(F90) -o $@ $^
	cp fl2xyz $(BIN)/fl2xyz

pc2xyz: poscar2bs.o
	$(F90) -o $@ $^
	cp pc2xyz $(BIN)/pc2xyz

all: $(OBJECTS)
	$(F90) -o tbfit $^ $(BLAS) $(LAPACK) $(SCALAPACK_LIB) $(SPGLIB_) $(INCLUDE)
	$(F90) -o pc2xyz poscar2bs.f90
	cp tbfit $(BIN)
	cp pc2xyz $(BIN)

touch:
	touch tbfit.f90

clean:
	rm $(BIN)/tbfit *.o *.mod

clean_pc2xyz:
	rm $(BIN)/pc2xyz poscar2bs.o

clean_get_ldos:
	rm get_ldos
