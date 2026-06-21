// SPDX-License-Identifier: Apache-2.0
// crt_host_test.c — run the scalar kernel core on the HOST and print checksum + samples, so the
// Python golden model can confirm the C is bit-exact before it ever runs on the x280.
//   cc -O2 crt_host_test.c -o /tmp/crt_host && /tmp/crt_host [shiftadd]
#include <stdio.h>
#include <stdlib.h>
#include <crt_kernel.h>

int main(int argc, char **argv) {
    static crt_u8 Aw[CRT_N][CRT_N], Ad[CRT_N][CRT_N];
    static crt_u16 C[CRT_N][CRT_N];
    int shiftadd = (argc > 1) ? atoi(argv[1]) : 0;
    crt_pattern(Aw, Ad);
    crt_matmul(Aw, Ad, C, shiftadd);
    printf("checksum %u\n", crt_checksum(C));
    printf("samples %u %u %u %u\n", C[0][0], C[1][2], C[15][15], C[31][31]);
    return 0;
}
