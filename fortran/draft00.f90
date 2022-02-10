PROGRAM MAIN
    REAL :: r_tmp0, r_tmp1
    REAL :: r_hf0
    INTEGER :: i_tmp0, i_tmp1
    r_hf0(x) = 233*x !must be at the first several line
    ! PRINT*, '# test READ PRINT, (input+233)'
    ! READ*, r_tmp0
    ! r_tmp0 = r_tmp0 + 233
    ! PRINT*, r_tmp0

    PRINT *, '# test operator'
    r_tmp0 = 1 + 2 - 3*4.0/5
    PRINT *, '(1 + 2 - 3*4.0/5)=', r_tmp0
    r_tmp0 = 0.3**3.0
    PRINT *, '(0.3**3.0)=', r_tmp0

    PRINT *, '# test anonymous function'
    PRINT *, 'hf0(2.0)=', r_hf0(2.0)
    PRINT *, 'hf0(3.0)=', r_hf0(3.0)

    PRINT *, '# test function'
    PRINT *, 'r_function0(2.0)=', r_function0(2.0)
    PRINT *, 'r_function0(3.0)=', r_function0(3.0)

    PRINT *, '# test math function'
    r_tmp0 = sin(0.233)**2 + cos(0.233)**2
    PRINT *, 'sin(0.233)**2+cos(0.233)**2=', r_tmp0
    r_tmp0 = asin(sin(0.233))
    PRINT *, 'asin(sin(0.233))=', r_tmp0

    PRINT *, '# test IF-THEN-ELSE-END'
    PRINT *, 'r_demo_if_else(2.0)=', r_demo_if_else(2.0)
    PRINT *, 'r_demo_if_else(-2.0)=', r_demo_if_else(-2.0)

    PRINT *, '# test subroutine'
    CALL hf_subroutine00(1.0, r_tmp0)
    PRINT *, 'hf_subroutine00(1.0, r_tmp0): ', r_tmp0

    ! abs(a).le.tiny(a)
END
!gfortran -o tbd00.exe draft00.f90

SUBROUTINE hf_subroutine00(r_x, r_ret)
    REAL:: r_x, r_ret
    r_ret = r_x + 233
END

FUNCTION r_function0(x)
    r_function0 = 233*x !return variable must be same as function name
    return
END


FUNCTION r_demo_if_else(x)
    IF (x > 0) THEN
        r_demo_if_else = x + 233
    ELSE
        r_demo_if_else = x - 233
    END IF
END
