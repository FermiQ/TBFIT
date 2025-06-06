#include "alias.inc"
subroutine plot_stm_image(PINPT, PGEOM, PKPTS, ETBA, flag_use_overlap)
   use parameters, only : incar, poscar, kpoints, energy
   use orbital_wavefunction, only : psi_rho
   use mpi_setup
   use time
   use memory
   use print_io
   implicit none
   type (incar)   :: PINPT
   type (poscar)  :: PGEOM
   type (kpoints) :: PKPTS
   type (energy)  :: ETBA
   character(*), parameter :: func = 'plot_stm_image'
   integer*4     mpierr
   integer*4     ix, iy, iz, i1, i2, i3, igrid
   integer*4     istm, ikk, iee, ie, is
   integer*4     nwrite, iline, nline, nresi
   integer*4     neig, ngrid, ng1, ng2, ng3
   integer*4     pid_stm_up, pid_stm_dn
   real*8        a1(3),a2(3),a3(3), vol
   real*8        origin(3,PGEOM%neig), origin_reset(3,PGEOM%neig)
   real*8        grid_a1(0:PGEOM%stm_ngrid(1)-1), rx(PGEOM%neig)
   real*8        grid_a2(0:PGEOM%stm_ngrid(2)-1), ry(PGEOM%neig)
   real*8        grid_a3(0:PGEOM%stm_ngrid(3)-1), rz(PGEOM%neig)
   complex*16    phi_r(PGEOM%neig)
   complex*16    psi_r_up_total(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3))
   complex*16    psi_r_dn_total(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3))
   complex*16    psi_r_up_(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3))
   complex*16    psi_r_dn_(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3))
   complex*16    psi_r_up(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3) )
   complex*16    psi_r_dn(PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3) )
   complex*16    V(PGEOM%neig*PINPT%ispin,PGEOM%nband*PINPT%nspin)
   complex*16    SV(PGEOM%neig*PINPT%ispin,PGEOM%nband*PINPT%nspin)
   character*8   corb(PGEOM%neig)
   integer*4     stm_erange(PGEOM%nband,PINPT%nspin), stm_neig(PINPT%nspin)
   character*2   spin_index_c(2)
   real*8        t0, t1
   character*4   timer
   real*8        zeff(PGEOM%neig)
   real*8        nqnum(PGEOM%neig)
   integer*4     lqnum(PGEOM%neig)
   character*2   orb(PGEOM%neig)
   logical       flag_use_overlap

   timer = 'init'
   spin_index_c(1) = 'up' 
   spin_index_c(2) = 'dn'

   write(message,*)' ' ; write_msg
   write(message,'(A)')' *- START PLOTTING: INTEGRATED CHARGE DENSITY (STM MODE)' ; write_msg
   call time_check(t1,t0,timer)
   
   call set_variable_plot_stm(PINPT, PGEOM, neig, ngrid, nwrite, nline, nresi, &
                              pid_stm_up, pid_stm_dn, vol, ng1, ng2, ng3, &
                              a1, a2, a3, origin, corb, grid_a1, grid_a2, grid_a3, &
                              zeff, nqnum, lqnum, orb)
#ifdef MPI
   call MPI_BARRIER(mpi_comm_earth, mpierr)
#endif
   if_main call report_memory(int8(ng1)*int8(ng2)*int8(ng3)*nprocs*PINPT%ispin*3, 16, 'Charge density') ! psi_r_up/dn, psi_r_up_/dn_, _up/dn_total
   if_main call report_memory(int8(PGEOM%neig)*int8(PINPT%ispin)*int8(PGEOM%nband)*int8(PINPT%nspin)*int8(nprocs)+&
                              int8(PGEOM%neig)*int8(PINPT%ispin)*int8(PGEOM%nband)*int8(PINPT%nspin)*int8(PKPTS%nkpoint), 16, 'Eigen vectors ') ! V*nprocs + ETBA%V(root)
 stm: do istm = 1, PINPT%n_stm
       psi_r_up_total = 0d0
       psi_r_dn_total = 0d0

       if_main call print_CHGCAR_stm_head(PINPT, PGEOM, pid_stm_up, pid_stm_dn, istm)
#ifdef MPI
       call MPI_Barrier(mpi_comm_earth,mpierr)
#endif

   kp: do ikk = 1, PKPTS%nkpoint
         call initialize_psi_r_stm(psi_r_up,psi_r_dn, ngrid,PINPT%ispin)
         stm_neig = 0; stm_erange = 0 ! initialize
         call get_stm_erange(PINPT, PKPTS, ETBA%E(:,ikk), neig, PGEOM%init_erange, PGEOM%nband, stm_neig, stm_erange, istm, ikk)
   spin: do is = 1, PINPT%nspin ! 2 for collinear,  1 for nonmag and non-collinear
           if_main call print_kpoint_index_info_header(stm_neig, PINPT%nspin, is, ikk, PKPTS%kpoint_reci(:,ikk))
           if_main call print_band_index_info_header(stm_neig, stm_erange, PINPT%nspin, is, PGEOM%nband)

           if(stm_neig(is) .gt. 0) then
#ifdef MPI
             if_main V=ETBA%V(:,:,ikk)
             call MPI_BCAST(V, size(V), MPI_COMPLEX16, 0, mpi_comm_earth, mpierr)
             if(flag_use_overlap) then
               if_main SV=ETBA%SV(:,:,ikk)
               call MPI_BCAST(SV, size(SV), MPI_COMPLEX16, 0, mpi_comm_earth, mpierr)
             endif
#else
             V=ETBA%V(:,:,ikk)
             if(flag_use_overlap) then
               SV=ETBA%SV(:,:,ikk)
             endif
#endif

#ifdef MPI
           ! THIS MPI ROUTINE ONLY WORKS FOR NON-SPIN_POLARIZED SYTEMS IN CURRENT VERSION: 2018 July 2 KHJ.
           ! It is due to the V(:,iee) is only for "up" part. For "dn" part, V(:,iee + nband)
           ! Note: I think above statement is obsolote as of 2020.
           !       However, one should carefully check it again. 2021. March 08 KHJ.
           call MPI_Barrier(mpi_comm_earth,mpierr)
#endif
     band: do ie = 1, stm_neig(is)
             iee = stm_erange(ie,is) - PGEOM%init_erange + 1
    cell_z: do iz =  -PINPT%repeat_cell_orb_plot(3),PINPT%repeat_cell_orb_plot(3) !ad hoc ... for MoTe2 grain boundary only... !WARNING!!
    cell_y: do iy =  -PINPT%repeat_cell_orb_plot(2),PINPT%repeat_cell_orb_plot(2) !ad hoc ... for MoTe2 grain boundary only... !WARNING!!
    cell_x: do ix =  -PINPT%repeat_cell_orb_plot(1),PINPT%repeat_cell_orb_plot(1) !ad hoc ... for MoTe2 grain boundary only... !WARNING!!
              call reset_orbital_origin(origin_reset, origin, neig, a1, a2, a3, ix, iy, iz)
      !grid_z: do i3=0,ng3-1
      grid_z: do i3=0+myid,ng3-1, nprocs
      grid_y: do i2=0,ng2-1
      grid_x: do i1=0,ng1-1
                igrid = i1+1+i2*ng1+i3*ng1*ng2
                call get_rxyz(rx,ry,rz, grid_a1, grid_a2, grid_a3, origin_reset, neig, ngrid, a1, a2, a3, i1,i2,i3)
                call get_orbital_wavefunction_phi_r(phi_r, rx,ry,rz, corb, neig, PINPT%rcut_orb_plot, .false., zeff, nqnum, lqnum, orb)
                if(PINPT%nspin .eq. 1) then ! non-mag or noncollinear
                  call get_psi_r_stm(psi_r_up(igrid),psi_r_dn(igrid),neig,PINPT%ispin,phi_r,V(:,iee),SV(:,iee), &
                                     is,PINPT%ispinor,PINPT%nspin, flag_use_overlap)
                elseif(PINPT%nspin .eq. 2) then ! collinear
                  call get_psi_r_stm(psi_r_up(igrid),psi_r_dn(igrid),neig,PINPT%ispin,phi_r,V(:,(/iee,iee+PGEOM%nband/)),SV(:,(/iee,iee+PGEOM%nband/)), &
                                     is,PINPT%ispinor,PINPT%nspin, flag_use_overlap)
                endif
              enddo grid_x
              enddo grid_y
              enddo grid_z
            enddo cell_x
            enddo cell_y
            enddo cell_z

           enddo band
           endif

#ifdef MPI
           ! THIS MPI ROUTINE ONLY WORKS FOR NON-SPIN_POLARIZED SYTEMS IN CURRENT VERSION: 2018 July 2 KHJ.
           ! Above warning should be removed.. but not has not been checked.. : 2018 Apr. 2 KHJ
!          call MPI_Allreduce(psi_r_up(:,ikk), psi_r_up_(:,ikk), size(psi_r_up_(:,ikk)), MPI_REAL8, MPI_SUM, mpi_comm_earth, mpierr)
           call MPI_Barrier(mpi_comm_earth,mpierr)
           if(is .eq. 1) then
             call MPI_Allreduce(psi_r_up, psi_r_up_, size(psi_r_up_), MPI_COMPLEX16, MPI_SUM, mpi_comm_earth, mpierr)
             psi_r_up_total = psi_r_up_total + psi_r_up_
             if(PINPT%ispinor .eq. 2) then
               call MPI_Allreduce(psi_r_dn, psi_r_dn_, size(psi_r_dn_), MPI_COMPLEX16, MPI_SUM, mpi_comm_earth, mpierr)
               psi_r_dn_total = psi_r_dn_total + psi_r_dn_
             endif
           elseif(is .eq. 2) then
             call MPI_Allreduce(psi_r_dn, psi_r_dn_, size(psi_r_dn_), MPI_COMPLEX16, MPI_SUM, mpi_comm_earth, mpierr)
             psi_r_dn_total = psi_r_dn_total + psi_r_dn_
           endif
           call MPI_Barrier(mpi_comm_earth,mpierr)
#else
           if(is .eq. 1) then
             psi_r_up_total = psi_r_up_total + psi_r_up
             if(PINPT%ispinor .eq. 2) then
               psi_r_dn_total = psi_r_dn_total + psi_r_dn
             endif
           elseif(is .eq. 2) then
             psi_r_dn_total = psi_r_dn_total + psi_r_dn
           endif
#endif
         enddo spin
       enddo kp

!      do ikk = 1, PKPTS%nkpoint
!        psi_r_up_total(:) = psi_r_up_total(:) + psi_r_up(:,ikk)
!        psi_r_dn_total(:) = psi_r_dn_total(:) + psi_r_dn(:,ikk)
!      enddo

       ! we write spin-dependent STM data if collinear. 
       ! if someone want to plot total STM (psi_r_up -> psi_r_up + psi_r_dn),
       if_main call write_rho_main_stm(pid_stm_up, pid_stm_dn, ngrid, nline, nwrite, nresi, psi_r_up_total, psi_r_dn_total, &
                                   PINPT%ispin, PINPT%nspin, PINPT%ispinor, .false.)
     enddo stm

   call time_check(t1,t0)
   write(message,*)' ' ; write_msg
   write(message,'(A,F10.4,A)')'   TIME for STM PLOT : ',t1, ' (sec)' ; write_msg
   write(message,'(A)')'*- END PLOTTING: STM PLOT' ; write_msg
   return
endsubroutine

subroutine write_rho_main_stm(pid_chg_up, pid_chg_dn, ngrid, nline, nwrite, nresi, psi_r_up, psi_r_dn, &
                          ispin, nspin, ispinor, flag_plot_wavefunction)
   implicit none
   integer*4      ispin, ispinor, nspin
   integer*4      pid_chg_up, pid_chg_dn, ngrid, nline, nwrite, nresi
   complex*16     psi_r_up(ngrid), psi_r_dn(ngrid)
   logical        flag_plot_wavefunction

   ! write rho (nonmag, noncol), rho_up (collinear), psi_up.majority (collin,noncollin)
   if(    ispinor .eq. 2 .and. .not. flag_plot_wavefunction) then   ! for noncol (rho)
     call print_PARCHG_main(pid_chg_up, ngrid, nline, nwrite, nresi, psi_r_up+psi_r_dn, .false.)
   elseif(ispinor .eq. 2 .and.       flag_plot_wavefunction) then   ! for noncol (psi)
     call print_PARCHG_main(pid_chg_up, ngrid, nline, nwrite, nresi, psi_r_up         , .true. )
     call print_PARCHG_main(pid_chg_dn, ngrid, nline, nwrite, nresi, psi_r_dn         , .true. )

   elseif(nspin   .eq. 2                                   ) then   ! for collin (rho or psi)
     call print_PARCHG_main(pid_chg_up, ngrid, nline, nwrite, nresi, psi_r_up,flag_plot_wavefunction)
     call print_PARCHG_main(pid_chg_dn, ngrid, nline, nwrite, nresi, psi_r_dn,flag_plot_wavefunction)

   elseif(ispin   .eq. 1                                   ) then   ! for nonmag (rho or psi)
     call print_PARCHG_main(pid_chg_up, ngrid, nline, nwrite, nresi, psi_r_up,flag_plot_wavefunction)
   endif


   return
endsubroutine

subroutine get_psi_r_stm(psi_r_up,psi_r_dn,nbasis,ispin,phi_r,V,SV,is,ispinor,nspin, flag_use_overlap)
   use parameters, only : energy
   use orbital_wavefunction, only: psi_rho
   implicit none
   type(energy) :: ETBA
   integer*4    igrid, nbasis, ispin, nspin
   integer*4    iee, ikk, is, ispinor
   complex*16   psi_r_up, psi_r_dn
   complex*16   phi_r(nbasis)
   complex*16   V(nbasis*ispin,nspin)
   complex*16   SV(nbasis*ispin,nspin)
   logical      flag_use_overlap

   if    (is .eq. 1 .and. ispinor .eq. 1) then
     psi_r_up = psi_r_up + psi_rho(phi_r, nbasis, ispin, V(:,1),SV(:,1), .false., 'up', flag_use_overlap)
   elseif(is .eq. 1 .and. ispinor .eq. 2) then
     psi_r_up = psi_r_up + psi_rho(phi_r, nbasis, ispin, V(:,1),SV(:,1), .false., 'up', flag_use_overlap)
     psi_r_dn = psi_r_dn + psi_rho(phi_r, nbasis, ispin, V(:,1),SV(:,1), .false., 'dn', flag_use_overlap)
   elseif(is .eq. 2 .and. ispinor .eq. 1) then
     psi_r_dn = psi_r_dn + psi_rho(phi_r, nbasis, ispin, V(:,2),SV(:,2), .false., 'dn', flag_use_overlap)
   endif

   return
endsubroutine

subroutine initialize_psi_r_stm(psi_r_up,psi_r_dn, ngrid, ispin)
   implicit none
   integer*4    ispin, ngrid, nkpoint
   complex*16   psi_r_up(ngrid), psi_r_dn(ngrid)
   logical      flag_plot_wavefunction

   psi_r_up = (0d0,0d0)

   if( ispin .eq. 2) psi_r_dn = (0d0,0d0)

   return
endsubroutine

subroutine print_CHGCAR_stm_head(PINPT, PGEOM, pid_stm_up, pid_stm_dn, istm)
   use parameters, only : incar, poscar
   use print_io
   use mpi_setup
   implicit none
   type(incar) :: PINPT
   type(poscar):: PGEOM
   integer*4      istm  
   integer*4      pid_stm_up, pid_stm_dn
   character*3    c_up,c_dn
   character*10   c_extension_up,c_extension_dn

   c_up = '' ; if_nsp2 c_up = '-up'
   c_dn = '' ; if_nsp2 c_dn = '-dn'

       if(istm .lt. 10) then 
         write(c_extension_up, '(A,I1,A)') '-STM-',istm,trim(c_up)
 if_nsp2 write(c_extension_dn, '(A,I1,A)') '-STM-',istm,trim(c_dn)
       elseif(istm .ge. 10) then
         write(c_extension_up, '(A,I2,A)') '-STM-',istm,trim(c_up)
 if_nsp2 write(c_extension_dn, '(A,I2,A)') '-STM-',istm,trim(c_dn)
       elseif(istm .ge. 100) then
         write(message,*)" !!! TOO MANY STM PLOT REQUESTED !!! NMAX STM = 99" ; write_msg
         stop
       endif
 
         call CHGCAR_stm_head(pid_stm_up, istm, PINPT, PGEOM, c_extension_up)
 if_nsp2 call CHGCAR_stm_head(pid_stm_dn, istm, PINPT, PGEOM, c_extension_dn)

   return
endsubroutine

subroutine CHGCAR_stm_head(pid_stm_, istm, PINPT, PGEOM, c_extension)
   use parameters, only : incar, poscar
   implicit none
   type(incar) :: PINPT
   type(poscar):: PGEOM
   integer*4      i
   integer*4      istm
   integer*4      pid_stm_
   character*10   c_extension
   character*40   fname

   write(fname,'(A,A)')'CHGCAR',trim(c_extension)

   open(pid_stm_, file=trim(fname), status='unknown')

   write(pid_stm_, '(A,F9.4,A,F9.4,A)')'INTEGRATED CHARGE DENSITY. ENERGY WINDOW= ( ', &
                                        PINPT%stm_emin(istm),' :',PINPT%stm_emax(istm),' )'

  !write(pid_stm_,*)PGEOM%a_scale
   write(pid_stm_,*)" 1.0000 " ! since we have already rescaled the a_latt when we read poscar (in read_poscar.f90)
   write(pid_stm_,'(3F20.16)')PGEOM%a_latt(1:3,1)
   write(pid_stm_,'(3F20.16)')PGEOM%a_latt(1:3,2)
   write(pid_stm_,'(3F20.16)')PGEOM%a_latt(1:3,3)
   write(pid_stm_,*)PGEOM%c_spec(:)
   write(pid_stm_,*)PGEOM%i_spec(:)
   write(pid_stm_,'(A)') "Direct" ! only direct coordinate will be written
   do i = 1, PGEOM%n_atom
     write(pid_stm_,'(3F20.16)') PGEOM%a_coord(:,i)
   enddo
   write(pid_stm_,*)" "
   write(pid_stm_,'(1x,3I6)')PGEOM%stm_ngrid(1:3)

   return
endsubroutine

subroutine get_stm_erange(PINPT, PKPTS, E, neig, iband, nband, stm_neig, stm_erange, istm, ikk)
   use parameters, only : incar, poscar, kpoints, energy
   implicit none
   type (incar)   :: PINPT
   type (kpoints) :: PKPTS
   integer*4     ie, ii, ispin, ikk
   integer*4     istm, neig, iband, nband
   integer*4     stm_erange(nband,PINPT%nspin), stm_neig(PINPT%nspin)
   real*8        E(nband * PINPT%nspin)
   character*2   spin_index
   integer*4     feast_ne(PINPT%nspin)

   if(PINPT%flag_sparse) then
     feast_ne = PINPT%feast_ne(1:PINPT%nspin, ikk)
   endif

   do ispin = 1, PINPT%nspin
     select case (ispin)
       case(1) ! spin up
         ii = 0
         if(PINPT%flag_sparse) then
           do ie = 1, feast_ne(1)   ! valid for nonmagnetic and noncollinear case, and spin-up for collinear
             if(E(ie) .ge. PINPT%stm_emin(istm) .and. E(ie) .le. PINPT%stm_emax(istm)) then
               ii = ii + 1
               stm_erange(ii, 1) = ie + iband - 1
             endif
           enddo
         else
           do ie = 1, nband   ! valid for nonmagnetic and noncollinear case, and spin-up for collinear
             if(E(ie) .ge. PINPT%stm_emin(istm) .and. E(ie) .le. PINPT%stm_emax(istm)) then
               ii = ii + 1
               stm_erange(ii, 1) = ie + iband - 1
             endif
           enddo
         endif

         stm_neig(1) = ii ! total number of bands within energy window

       case(2) ! spin down
         ii = 0
         if(PINPT%flag_sparse .and. feast_ne(2) .ge. 1) then
           do ie = 1+nband, nband + feast_ne(2) ! this is only valid if nspin = 2 (collinear) case
             if(E(ie) .ge. PINPT%stm_emin(istm) .and. E(ie) .le. PINPT%stm_emax(istm)) then
               ii = ii + 1
               stm_erange(ii, 2) = ie + iband - 1
             endif
           enddo
         else
           do ie = 1+nband, nband*2 ! this is only valid if nspin = 2 (collinear) case
             if(E(ie) .ge. PINPT%stm_emin(istm) .and. E(ie) .le. PINPT%stm_emax(istm)) then
               ii = ii + 1
               stm_erange(ii, 2) = ie + iband - 1
             endif
           enddo
         endif

         stm_neig(2) = ii

     end select
   enddo
   return
endsubroutine

subroutine set_variable_plot_stm(PINPT, PGEOM, neig, ngrid, nwrite, nline, nresi, &
                                 pid_stm_up, pid_stm_dn, vol, ng1, ng2, ng3, &
                                 a1, a2, a3, origin, corb, grid_a1, grid_a2, grid_a3, zeff, nqnum, lqnum, orb)
   use parameters, only : incar, poscar, pid_stm
   use element_info, only: angular
   use print_io
   use mpi_setup
   implicit none
   type(incar)  ::  PINPT
   type(poscar) ::  PGEOM
   integer*4        i, iorbital, iatom, iorb
   integer*4        neig, ngrid, nwrite, nline, nresi
   integer*4        ng1, ng2, ng3
   integer*4        pid_stm_up, pid_stm_dn
   real*8           a1(3),a2(3),a3(3), a2xa3(3), vol
   real*8           rshift(3)
   character(*), parameter :: func = 'set_variable_plot_stm'
   character*8      corb(PGEOM%neig)
   real*8           zeff(PGEOM%neig)
   real*8           nqnum(PGEOM%neig)
   integer*4        lqnum(PGEOM%neig)
   character*2      orb(PGEOM%neig), orb_
   real*8           origin(3,PGEOM%neig)
   real*8           grid_d1, grid_a1(0:PGEOM%stm_ngrid(1)-1)
   real*8           grid_d2, grid_a2(0:PGEOM%stm_ngrid(2)-1)
   real*8           grid_d3, grid_a3(0:PGEOM%stm_ngrid(3)-1)

   neig   = PGEOM%neig
   ngrid  = PGEOM%stm_ngrid(1)*PGEOM%stm_ngrid(2)*PGEOM%stm_ngrid(3)
   ng1    = PGEOM%stm_ngrid(1) ; ng2     = PGEOM%stm_ngrid(2) ; ng3     = PGEOM%stm_ngrid(3)
   nwrite = 5
   nline=int(ngrid/nwrite)
   nresi=mod(ngrid,nwrite)

   pid_stm_up = pid_stm
   pid_stm_dn = pid_stm + 10

!  rshift(1:3)=PINPT%r_origin(1:3) 
   rshift(1:3)=0d0 ! shift of origin is not assumed in STM plot yet... (need to be improved later)
   a1=PGEOM%a_latt(1:3,1)
   a2=PGEOM%a_latt(1:3,2)
   a3=PGEOM%a_latt(1:3,3)
   call vcross(a2xa3,a2,a3)
   vol=dot_product(a1,a2xa3)

   iorbital = 0
   do iatom = 1, PGEOM%n_atom
     do iorb = 1, PGEOM%n_orbital(iatom)
       iorbital = iorbital + 1
       origin(:,iorbital) =( PGEOM%a_coord(1,iatom) + rshift(1) )*a1(:) + &
                           ( PGEOM%a_coord(2,iatom) + rshift(2) )*a2(:) + &
                           ( PGEOM%a_coord(3,iatom) + rshift(3) )*a3(:)
       corb(iorbital)=trim(PGEOM%c_orbital(iorb,iatom))

       zeff(iorbital) = PGEOM%z_eff_nuc(iorb, iatom)
       nqnum(iorbital)= PGEOM%n_quantum(iatom)
       lqnum(iorbital)= PGEOM%l_quantum(iorb,iatom)
       write(orb(iorbital),'(I0,A)') PGEOM%orb_n_quantum(iorb,iatom), angular(PGEOM%l_quantum(iorb,iatom))
       !write(orb_,'(I0,A)') PGEOM%orb_n_quantum(iorb,iatom), angular(PGEOM%l_quantum(iorb,iatom))
       !orb(iorbital) = orb_

     enddo
   enddo
   if (iorbital .ne. PGEOM%neig) then
     write(message,'(A,A)')'  !WARNING! iorbital is not same as neig!, please check again. ',func ; write_msg
     stop
   endif

   grid_d1 = 1d0/dble(ng1)
   grid_d2 = 1d0/dble(ng2)
   grid_d3 = 1d0/dble(ng3)

   grid_a1 = (/(dble(i)*grid_d1, i=0, ng1-1)/)
   grid_a2 = (/(dble(i)*grid_d2, i=0, ng2-1)/)
   grid_a3 = (/(dble(i)*grid_d3, i=0, ng3-1)/)

   return
endsubroutine

subroutine print_kpoint_index_info_header(stm_neig, nspin, is, ikk, kpoint_reci)
   use mpi_setup
   use print_io
   implicit none
   integer*4   nspin, is, ikk
   integer*4   stm_neig(nspin)
   real*8      kpoint_reci(3)

   if( (sum(stm_neig(:)) .ge. 1) .and. is .eq. 1) then
     write(message,'(A,I5,A,3F9.4,A)')"   * K-POINT INDEX :",ikk,' (',kpoint_reci(:),') (reciprocal unit)' ; write_msg
    endif

   return
endsubroutine

subroutine print_band_index_info_header(stm_neig,stm_erange,nspin, is, nband)
   use mpi_setup
   use print_io
   implicit none
   integer*4   nspin, is, nband 
   integer*4   stm_neig(nspin), stm_erange(nband,nspin)

   if( stm_neig(is) .ge. 1 ) then

     if(nspin .eq. 2) then
       if(is .eq. 1) write(message,'(A,3x,9999I5)')"     -- BAND INDEX : (SPIN-UP)",stm_erange(1:stm_neig(is),is)
       if(is .eq. 2) write(message,'(A,3x,9999I5)')"     -- BAND INDEX : (SPIN-DN)",stm_erange(1:stm_neig(is),is)
       write_msg
     else
       write(message,'(A,3x,9999I5)')"     -- BAND INDEX : ",stm_erange(1:stm_neig(is),is) ; write_msg
     endif

   endif

   return
endsubroutine

subroutine set_ngrid(PINPT, PGEOM)
   use parameters, only : incar, poscar
   use mpi_setup
   use print_io
   implicit none
   type(incar)   :: PINPT
   type(poscar)  :: PGEOM
   integer*4        mpierr
   real*8           enorm
   external         enorm  

   if(PGEOM%ngrid(1) .eq. -1 .and. PINPT%flag_plot_eigen_state) then
     PGEOM%ngrid(1) = nint(enorm(3, PGEOM%a_latt(:,1)) / 0.1d0)
     PGEOM%ngrid(2) = nint(enorm(3, PGEOM%a_latt(:,2)) / 0.1d0)
     PGEOM%ngrid(3) = nint(enorm(3, PGEOM%a_latt(:,3)) / 0.1d0)
     PGEOM%ngrid(1) = PGEOM%ngrid(1) + mod(PGEOM%ngrid(1),2)
     PGEOM%ngrid(2) = PGEOM%ngrid(2) + mod(PGEOM%ngrid(2),2)
     PGEOM%ngrid(3) = PGEOM%ngrid(3) + mod(PGEOM%ngrid(3),2)
     write(message,'(A,3(I6))')'   N_GRID: (for EIGPLOT) ',PGEOM%ngrid(1:3) ; write_msg
   endif

   if(PGEOM%stm_ngrid(1) .eq. -1 .and. PINPT%flag_plot_stm_image) then
     PGEOM%stm_ngrid(1) = nint(enorm(3, PGEOM%a_latt(:,1)) / 0.1d0)
     PGEOM%stm_ngrid(2) = nint(enorm(3, PGEOM%a_latt(:,2)) / 0.1d0)
     PGEOM%stm_ngrid(3) = nint(enorm(3, PGEOM%a_latt(:,3)) / 0.1d0)
     PGEOM%stm_ngrid(1) = PGEOM%stm_ngrid(1) + mod(PGEOM%stm_ngrid(1),2)
     PGEOM%stm_ngrid(2) = PGEOM%stm_ngrid(2) + mod(PGEOM%stm_ngrid(2),2)
     PGEOM%stm_ngrid(3) = PGEOM%stm_ngrid(3) + mod(PGEOM%stm_ngrid(3),2)
     write(message,'(A,3(I6))')'   N_GRID: (for STMPLOT) ',PGEOM%stm_ngrid(1:3) ; write_msg
   endif
   
   return
endsubroutine
