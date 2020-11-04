module get_parameter
  use mpi_setup

  contains
    elemental subroutine get_param(PPRAM, param_index, isub, param)
       use parameters, only: params
       implicit none
       type(params),intent(in) :: PPRAM
       integer*4,   intent(in) :: param_index ! parameter index
       integer*4,   intent(in) :: isub  ! sub parameter index  
       integer*4                  k
       real*8,      intent(out):: param

       if(param_index .ne. 0) then
         if(PPRAM%slater_koster_type .gt. 10) then
           k = nint(PPRAM%param_const_nrl(4,1, param_index)) ! if fixed
           if(k .ge. 1) then
             param = PPRAM%param_const_nrl(5,isub,param_index) ! restore fixed value
           else
             param = PPRAM%param_nrl(isub,param_index)
           endif
         else
           k = nint(PPRAM%param_const(4, param_index))
           if(k .ge. 1) then
             param = PPRAM%param_const(5,param_index)
           else
             param = PPRAM%param(param_index)
           endif
         endif

       elseif(param_index .eq. 0) then
         param = 0d0
       endif

    return
    endsubroutine
endmodule
