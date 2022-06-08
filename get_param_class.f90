#include "alias.inc"
!subroutine get_nn_class(PGEOM, iatom,jatom,dij,onsite_tol, nn_class, r0)
subroutine get_nn_class(PGEOM, iatom,jatom,dij,onsite_tol, nn_class, r0, rc)
   use parameters, only : poscar
   real*8      r0  ! reference distance
   real*8      rc  ! cutoff    distance
   real*8      dij, onsite_tol
   integer*4   n, iatom, jatom, li, lj
   integer*4   ii, nn, max_nn, nn_class, i_dummy
   character*8 c_atom_i, c_atom_j
   real*8      d_range
   real*8      dummy
   real*8      dist_nn(0:100)
   real*8      r0_nn(0:100)
   type(poscar) :: PGEOM

   c_atom_i=adjustl(trim(PGEOM%c_spec( PGEOM%spec(iatom) )))
   li=len_trim(c_atom_i)
   c_atom_j=adjustl(trim(PGEOM%c_spec( PGEOM%spec(jatom) )))
   lj=len_trim(c_atom_j)
   
   max_nn = 0
   dist_nn(0)=onsite_tol
   do n = 1, PGEOM%n_nn_type
     index_iatom=index( PGEOM%nn_pair(n),c_atom_i(1:li) ) ! pick former. for example pick A out of "xx_xx_AB"
     index_jatom=index( PGEOM%nn_pair(n),c_atom_j(1:lj) ,.TRUE. ) ! pick later one. for example pick B out of "xx_xx_AB"

     if(index_iatom .eq. 0 .or. index_jatom .eq. 0) then
       cycle
     elseif(index_iatom .ge. 1 .and. index_jatom .ge. 1 .and. index_iatom .ne. index_jatom) then ! if correctly pick A and B
       nn = len_trim(PGEOM%nn_pair(n)) - li - lj
     else
       cycle
     endif

     if( nn .ge. 1) then
       max_nn = max(nn, max_nn)
       dist_nn(nn) = PGEOM%nn_dist(n)  ! cutoff distance for n-th nn neighbor pair
       r0_nn(nn)   = PGEOM%nn_r0(n)    ! reference distance R0 for n-th nn neighbor pair
                                       ! if SK_SCALE_TYPE >= 11, max(nn_r0) will be taken for the R0 to calculate rho_i
     endif
   enddo

   nn_class = -9999
   do n = 0, max_nn
     if( n .eq. 0 .and. dij .le. dist_nn(n) )then  ! check onsite
       nn_class = 0
       r0       = 0d0
       rc       = 0d0
       exit
     elseif ( n .ge. 1 .and. dij .gt. dist_nn(n - 1) .and. dij .le. dist_nn(n) ) then
       nn_class = n
       r0       = r0_nn(n)
       rc       = dist_nn(n)
       exit
     endif
   enddo

return
endsubroutine
subroutine get_param_class(PGEOM,iorb,jorb,iatom,jatom,param_class)
   use parameters, only : poscar
   implicit none
   character*8 c_orb_i, c_orb_j
   real*8      r0, dij, onsite_tol
   integer*4   n,iorb,jorb,iatom,jatom,li, lj
   integer*4   index_iatom,index_jatom
   integer*4   ii, nn, max_nn, nn_class,i_dummy
   character*2 param_class
   character*2 c_orb_ij
   character*8 c_atom_i, c_atom_j
   real*8      d_range
   real*8      dummy
   real*8      dist_nn(0:100)
   real*8      r0_nn(0:100)
   type(poscar) :: PGEOM

   ! set hopping property: ss, sp, pp, dd ... etc. ?
   c_orb_i=PGEOM%c_orbital(iorb,iatom)
   c_orb_j=PGEOM%c_orbital(jorb,jatom)
   write(c_orb_ij,'(A1,A1)')c_orb_i(1:1),c_orb_j(1:1)
   
   select case ( c_orb_ij )

     case('ss')
       param_class = 'ss'
     case('sp'      )
       param_class = 'sp'
     case('ps'      )
       param_class = 'ps'
     case('sd'      )
       param_class = 'sd'
     case('ds'      )
       param_class = 'ds'
     case('sf'      )
       param_class = 'sf'
     case('fs'      )
       param_class = 'fs'
     case('pp')
       param_class = 'pp'
     case('pd'      )
       param_class = 'pd'
     case('dp'      )
       param_class = 'dp'
     case('pf'      )
       param_class = 'pf'
     case('fp'      )
       param_class = 'fp'
     case('dd')
       param_class = 'dd'
     case('df'      )
       param_class = 'df'
     case('fd'      )
       param_class = 'fd'
     case('ff')
       param_class = 'ff'
     case('cc')
       param_class = 'cc' ! user defined (for lattice model, BiSi110 example)
     case('xx')
       param_class = 'xx' ! user defined (for sk-type model, TaS2 example..)
   end select
 
return
endsubroutine
function param_class_rev(param_class)
   implicit none
   character*2  param_class
   character*2  param_class_rev

   param_class_rev=param_class(1:1)//param_class(2:2)

   return
endfunction
subroutine get_onsite_param_index(ionsite_param_index, PPRAM, ci_orb, cj_orb, c_atom)
   use parameters, only : params
   use print_io
   use mpi_setup
   implicit none
   type(params) :: PPRAM
   integer*4    i, lio, ljo, la, mpierr
   integer*4    ionsite_param_index
   character*8  ci_orb, cj_orb, c_atom
   character*20 ee_name

   lio = len_trim(ci_orb)
   ljo = len_trim(cj_orb)
   la = len_trim(c_atom)
   write(ee_name,*)'e_',ci_orb(1:lio),'_',c_atom(1:la)
   ionsite_param_index = 0

   if(ci_orb(1:lio) .eq. cj_orb(1:ljo)) then
    call get_param_index(PPRAM, ee_name, ionsite_param_index)
   endif

   if ( ionsite_param_index .eq. -1) then
     write(message,'(A,A,A)')'    !WARNING! Onsite energy for ', adjustl(trim(ee_name)), &
                       ' is not asigned. Please check "SET TBPARAM" tag. Exit...' ; write_msg
     stop
   endif
   
return
endsubroutine

subroutine get_sk_index_set(index_sigma,index_pi,index_delta, &
                            index_sigma_scale,index_pi_scale,index_delta_scale, &
                            PPRAM, param_class, nn_class, &
                            PGEOM, i_atom, j_atom, &
                            ci_site, cj_site, flag_use_site_cindex, flag_use_overlap)
   use parameters, only : params, poscar
   implicit none
   type(params)            ::  PPRAM
   type(poscar)            ::  PGEOM 
   integer*4                   nn_class
   integer*4                   imode
   integer*4                   i, lia, lja, lp
   integer*4                   i_atom, j_atom ! atom number as appeared in GFILE
   integer*4                   index_sigma,index_pi,index_delta
   integer*4                   index_sigma_scale,index_pi_scale,index_delta_scale
   character*16                cij_pair
   character*2                 param_class
   character*8                 ci_atom , cj_atom
   character*20,intent(in) ::  ci_site , cj_site 
   character*28            ::  ci_atom_, cj_atom_
   logical      flag_scale, flag_use_site_cindex, flag_use_overlap, flag_equiv

   ! initialize  
   index_sigma       =  0
   index_pi          =  0
   index_delta       =  0
   index_sigma_scale =  0
   index_pi_scale    =  0
   index_delta_scale =  0
   flag_equiv        =  .false.
   imode             =  0
   ci_atom           =  PGEOM%c_spec(PGEOM%spec(i_atom))
   cj_atom           =  PGEOM%c_spec(PGEOM%spec(j_atom))
   if( PGEOM%spec_equiv(i_atom) .eq. PGEOM%spec_equiv(j_atom)) flag_equiv = .true.

   if( .not. flag_equiv ) then
 lp1:do i=1,PGEOM%n_orbital(i_atom)
       if(param_class(1:1) .eq. PGEOM%c_orbital(i,i_atom)(1:1)) then
         imode = imode + 1
         exit lp1 ! exit loop if find orbital 1 in atom i
       endif
     enddo lp1
 lp2:do i=1,PGEOM%n_orbital(i_atom)
       if(param_class(2:2) .eq. PGEOM%c_orbital(i,i_atom)(1:1)) then
         imode = imode + 1
         exit lp2 ! exit loop if find orbital 2 in atom i
       endif
     enddo lp2
 lp3:do i=1,PGEOM%n_orbital(j_atom)
       if(param_class(1:1) .eq. PGEOM%c_orbital(i,j_atom)(1:1)) then
         imode = imode + 1
         exit lp3 ! exit loop if find orbital 1 in atom j
       endif
     enddo lp3
 lp4:do i=1,PGEOM%n_orbital(j_atom)
       if(param_class(2:2) .eq. PGEOM%c_orbital(i,j_atom)(1:1)) then
         imode = imode + 1
         exit lp4 ! exit loop if find orbital 2 in atom j
       endif
     enddo lp4
   else
     imode = 0
   endif
   ! ab_ij (a,b : orbital class (s, p, d..), i, j : atom index(atom i and j)
   ! imode = 0 : no orbital found in atom i and j          -> no matching orbital 
   ! imode = 1 : ex : sp_WH  =>  Tungsten:d, Hydrogen:s)   -> no matching orbital
   ! imode = 2 : ex : sd_WSe =>  Selenium:s, Tungsten:d)   -> 1  matching orbital
   ! imode = 3 : ex : sp_SeW =>  Selenium:sp, Tungsten:sd) -> 1  matching orbital
   ! imode = 4 : ex : sp_CSi =>  Carbon:sp, Silicon:sp)    -> 2  matching orbital
   !                  sp_SiC =>  Carbon:sp, Silicon:sp)       we need to distinguish this case

   if(.not.flag_use_site_cindex) then

     flag_scale = .false.
     call get_param_name_index(PPRAM, param_class, 'sigma', nn_class, ci_atom, cj_atom, flag_scale, index_sigma      , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'pi'   , nn_class, ci_atom, cj_atom, flag_scale, index_pi         , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'delta', nn_class, ci_atom, cj_atom, flag_scale, index_delta      , flag_use_overlap, imode)

     flag_scale = .true. 
     call get_param_name_index(PPRAM, param_class, 'sigma', nn_class, ci_atom, cj_atom, flag_scale, index_sigma_scale, flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'pi'   , nn_class, ci_atom, cj_atom, flag_scale, index_pi_scale   , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'delta', nn_class, ci_atom, cj_atom, flag_scale, index_delta_scale, flag_use_overlap, imode)

   elseif(flag_use_site_cindex) then

     write(ci_atom_,'(A,A)')trim(ci_atom),trim(ci_site)
     write(cj_atom_,'(A,A)')trim(cj_atom),trim(cj_site)

     flag_scale = .false.
     call get_param_name_index(PPRAM, param_class, 'sigma', nn_class, ci_atom_, cj_atom_, flag_scale, index_sigma      , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'pi'   , nn_class, ci_atom_, cj_atom_, flag_scale, index_pi         , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'delta', nn_class, ci_atom_, cj_atom_, flag_scale, index_delta      , flag_use_overlap, imode)

     flag_scale = .true.
     call get_param_name_index(PPRAM, param_class, 'sigma', nn_class, ci_atom_, cj_atom_, flag_scale, index_sigma_scale, flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'pi'   , nn_class, ci_atom_, cj_atom_, flag_scale, index_pi_scale   , flag_use_overlap, imode)
     call get_param_name_index(PPRAM, param_class, 'delta', nn_class, ci_atom_, cj_atom_, flag_scale, index_delta_scale, flag_use_overlap, imode)

   endif

return
endsubroutine
subroutine get_local_U_param_index(local_U_param_index, PPRAM, nn_class, param_class, ci_atom)
   use parameters, only : params
   implicit none
   type(params) :: PPRAM
   integer*4       nn_class
   integer*4       i, ii, liatom, lparam
   integer*4       local_U_param_index
   character*8     ci_atom
   character*2     param_class, param_class_
   character*40    local_U_param_name 


   !initialize
   local_U_param_index = 0

   liatom = len_trim(ci_atom)
   lparam = len_trim(param_class)
   param_class_ = adjustl(trim(param_class))

   if( nn_class .eq. 0) then
     if( adjustl(trim(param_class_)) .eq. 'ss' ) write(local_U_param_name ,  99)'local_U_s_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'pp' ) write(local_U_param_name ,  99)'local_U_p_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'dd' ) write(local_U_param_name ,  99)'local_U_d_',ci_atom(1:liatom)

     ! user defined param class and its corresponding parameter type. Here, in this case (TaS2), xp1, xp2, xp3 
     ! orbitals are used so that the local_U index follows local_U_'x'_ someting.
     if( adjustl(trim(param_class_)) .eq. 'xx' ) write(local_U_param_name ,  99)'local_U_x_',ci_atom(1:liatom)

     ! user defined param class and its corresponding parameter type. Here, in this case (BiSi110), cp1 orbital
     ! orbitals are used so that the local_U index follows local_U_'c'_ someting.
     if( adjustl(trim(param_class_)) .eq. 'cc' ) write(local_U_param_name ,  99)'local_U_c_',ci_atom(1:liatom)
   endif

   call get_param_index(PPRAM, local_U_param_name, local_U_param_index)
99 format(A,A)   

return
endsubroutine
subroutine get_l_onsite_param_index(l_onsite_param_index, PPRAM, ci_atom)
   use parameters, only : params
   implicit none
   type(params) :: PPRAM
   integer*4       i, ii, liatom, lparam
   integer*4       l_onsite_param_index
   character*8     ci_atom
   character*40    l_onsite_param_name
  
   ! initialize
   l_onsite_param_index = 0
   
   liatom = len_trim(ci_atom)

   write(l_onsite_param_name, 99)'lonsite_',ci_atom(1:liatom)
   
   call get_param_index(PPRAM, l_onsite_param_name, l_onsite_param_index)

99 format(A,A)
   
return
endsubroutine
subroutine get_plus_U_param_index(plus_U_param_index, PPRAM, nn_class, param_class, ci_atom)
   use parameters, only : params
   implicit none
   type(params) :: PPRAM
   integer*4       nn_class
   integer*4       i, ii, liatom, lparam
   integer*4       plus_U_param_index
   character*8     ci_atom
   character*2     param_class, param_class_
   character*40    plus_U_param_name

   !initialize
   plus_U_param_index = 0
   
   liatom = len_trim(ci_atom)
   lparam = len_trim(param_class)
   param_class_ = adjustl(trim(param_class))

   if( nn_class .eq. 0 ) then
     if( adjustl(trim(param_class_)) .eq. 'ss' ) write(plus_U_param_name ,  99)'plus_U_s_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'pp' ) write(plus_U_param_name ,  99)'plus_U_p_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'dd' ) write(plus_U_param_name ,  99)'plus_U_d_',ci_atom(1:liatom)
   endif

   call get_param_index(PPRAM, plus_U_param_name, plus_U_param_index)

99 format(A,A)

return
endsubroutine
subroutine get_stoner_I_param_index(stoner_I_param_index, PPRAM, nn_class, param_class, ci_atom)
   use parameters, only : params
   implicit none
   type(params) :: PPRAM
   integer*4       nn_class
   integer*4       i, ii, liatom, lparam
   integer*4       stoner_I_param_index
   character*8     ci_atom
   character*2     param_class, param_class_
   character*40    stoner_I_param_name


  !initialize
  stoner_I_param_index = 0

   liatom = len_trim(ci_atom)
   lparam = len_trim(param_class)
   param_class_ = adjustl(trim(param_class))

   if( nn_class .eq. 0) then
     if( adjustl(trim(param_class_)) .eq. 'ss' ) write(stoner_I_param_name ,  99)'stoner_I_s_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'pp' ) write(stoner_I_param_name ,  99)'stoner_I_p_',ci_atom(1:liatom)
     if( adjustl(trim(param_class_)) .eq. 'dd' ) write(stoner_I_param_name ,  99)'stoner_I_d_',ci_atom(1:liatom)

     ! for TaS2, user defined xx-class
     if( adjustl(trim(param_class_)) .eq. 'xx' ) write(stoner_I_param_name ,  99)'stoner_I_x_',ci_atom(1:liatom)

     ! for Bi/Si110, user defined cc-class
     if( adjustl(trim(param_class_)) .eq. 'cc' ) write(stoner_I_param_name ,  99)'stoner_I_c_',ci_atom(1:liatom)

   endif

   call get_param_index(PPRAM, stoner_I_param_name, stoner_I_param_index)

99 format(A,A)

return
endsubroutine
subroutine get_param_name(param_name, param_class, param_type, nn_class, ci_atom, cj_atom, flag_scale, flag_use_overlap)
   implicit none
   integer*4    lia, lja, lp
   integer*4    nn_class
   character*2  param_class
   character*8  param_type
   character*40 param_name
   logical      flag_scale, flag_use_overlap
   character(*), intent(in) ::  ci_atom, cj_atom

   lia = len_trim(ci_atom)
   lja = len_trim(cj_atom)
   lp  = len_trim(param_class)

   if(.not. flag_use_overlap) then
     if(.not. flag_scale) then
       if(nn_class .lt. 10) then
         write(param_name,99)      param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       elseif(nn_class .ge. 10) then
         write(param_name,98)      param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       endif
     elseif(flag_scale) then
       if(nn_class .lt. 10) then
         write(param_name,89) 's_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       elseif(nn_class .ge. 10) then
         write(param_name,88) 's_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       endif
     endif
   elseif( flag_use_overlap) then
     if(.not. flag_scale) then
       if(nn_class .lt. 10) then
         write(param_name,89) 'o_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       elseif(nn_class .ge. 10) then
         write(param_name,88) 'o_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       endif
     elseif(flag_scale) then
       if(nn_class .lt. 10) then
         write(param_name,89)'os_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       elseif(nn_class .ge. 10) then
         write(param_name,88)'os_',param_class(1:lp),param_type(1:1),'_',nn_class,'_',ci_atom(1:lia),cj_atom(1:lja)
       endif
     endif
   endif

99 format(3A,I1,3A)
89 format(4A,I1,3A)
98 format(3A,I2,3A)
88 format(4A,I2,3A)

return
endsubroutine

subroutine get_param_name_index(PPRAM, param_class, param_type, nn_class, ci_atom, cj_atom, &
                                flag_scale, param_index, flag_use_overlap, imode)
   use parameters, only : params
   implicit none
   type(params)            :: PPRAM
   integer*4                  i_atempt
   integer*4                  imode
   integer*4                  nn_class
   integer*4                  param_index
   character*2                param_class
   character*2                param_class_
   character*40               param_name
   logical                    flag_scale, flag_use_overlap
   character(*),intent(in) :: ci_atom, cj_atom
   character(*),intent(in) :: param_type

   if(imode .ne. 4) then

loop1:do i_atempt = 0, 1
       if(i_atempt .eq. 0) call get_param_name(param_name, param_class, trim(param_type), nn_class, ci_atom, cj_atom, flag_scale, flag_use_overlap)
       if(i_atempt .eq. 1) call get_param_name(param_name, param_class, trim(param_type), nn_class, cj_atom, ci_atom, flag_scale, flag_use_overlap)
       call get_param_index(PPRAM, param_name, param_index)
       if(param_index .gt. 0) exit loop1 
     enddo loop1

     if(param_index .eq. 0) then
       param_class_ = param_class(2:2)//param_class(1:1)
 loop2:do i_atempt = 0, 1
         if(i_atempt .eq. 0) call get_param_name(param_name, param_class_, trim(param_type), nn_class, ci_atom, cj_atom, flag_scale, flag_use_overlap)
         if(i_atempt .eq. 1) call get_param_name(param_name, param_class_, trim(param_type), nn_class, cj_atom, ci_atom, flag_scale, flag_use_overlap)
         call get_param_index(PPRAM, param_name, param_index)
         if(param_index .gt. 0) exit loop2
       enddo loop2
     endif
   elseif(imode .eq. 4) then
     ! sp_AB where A and B has either s and p
     ! if A != B parameter should be distinguished. => sp_AB != sp_BA
     ! if A  = B parameter can be defined without distinguish => sp_AA = ps_AA
     call get_param_name(param_name, param_class, trim(param_type), nn_class, ci_atom, cj_atom, flag_scale, flag_use_overlap)
     call get_param_index(PPRAM, param_name, param_index)
     if(param_index .gt. 0) return
     param_class_ = param_class(2:2)//param_class(1:1)
     call get_param_name(param_name, param_class_, trim(param_type), nn_class, cj_atom, ci_atom, flag_scale, flag_use_overlap)
     call get_param_index(PPRAM, param_name, param_index)
   endif

   return
endsubroutine
subroutine get_param_index(PPRAM, param_name, param_index)
   use parameters, only : params
   implicit none
   type(params) :: PPRAM
   integer*4       i, k
   integer*4       param_index
   character*40    pname_file, pname 
   character*40    str2lowcase
   external     :: str2lowcase
   character(*), intent(in) ::  param_name

   pname = adjustl(trim(param_name))
   pname = str2lowcase(pname)
   param_index = 0
   ! find parameter index with given parameter name
   do i = 1, PPRAM%nparam
     pname_file=adjustl(trim(PPRAM%param_name(i)))
     pname_file=str2lowcase(pname_file)

     if( trim(pname_file) .eq. adjustl(trim(pname)) ) then
       if( PPRAM%slater_koster_type .gt. 10) then
         k = nint(PPRAM%param_const_nrl(1,1,i))
       else
         k = nint(PPRAM%param_const(1,i))
       endif

       if( k .ge. 1 ) then
         param_index = k  ! set constraint condition: if equal to
       else
         param_index = i
       endif

       exit
     endif
   enddo

return
endsubroutine
