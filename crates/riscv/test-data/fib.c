void _start() {
    unsigned long a = 0, b = 1;
    for (int i = 0; i < 32768; i++) {
        unsigned long t = a + b;
        a = b;
        b = t;
    }
    register unsigned long result __asm__("a0") = b;
    __asm__ volatile("ecall" : : "r"(result));
    __builtin_unreachable();
}
