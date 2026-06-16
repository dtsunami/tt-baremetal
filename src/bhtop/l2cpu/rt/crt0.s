# bhtop.l2cpu C runtime: _start for freestanding C. Sets sp, zeroes .bss, calls
# main(), then parks. Linked first (.text._start) so it sits at LOAD_ADDR. With this,
# a C "kernel" is just:  int main(void) { ... }
    .option norvc
    .section .text._start, "ax", @progbits
    .globl _start
    .type  _start, @function
_start:
    la      sp, __stack_top          # absolute (no-PIC, --no-relax)
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f               # zero bss (8-byte aligned by link.ld)
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  call    main
3:  wfi                              # main returned -> park
    j       3b
    .size _start, .-_start
