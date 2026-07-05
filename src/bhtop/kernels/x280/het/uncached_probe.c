// x280 writes distinct sentinels across the GDDR data window; Tensix NoC-reads each to find which
// addresses are UNCACHED (x280 write reached GDDR) vs cached (stale).
#include <stdint.h>
int main(void){
    ((volatile uint32_t*)0x30002400u)[0]=0xA0000000u;
    ((volatile uint32_t*)0x30002800u)[0]=0xA0000001u;
    ((volatile uint32_t*)0x30003800u)[0]=0xA0000002u;
    ((volatile uint32_t*)0x30005000u)[0]=0xA0000003u;
    ((volatile uint32_t*)0x30007000u)[0]=0xA0000004u;
    ((volatile uint32_t*)0x30002000u)[0]=0xA0000005u;   // tele (known uncached) — control
    for(;;){}
}
