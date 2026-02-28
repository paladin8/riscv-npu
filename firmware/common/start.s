    .section .text.init
    .globl _start
_start:
    la      sp, _stack_top
    call    main
    # main returned; a0 holds return value
    li      a7, 93          # exit syscall number (convention)
    ecall
