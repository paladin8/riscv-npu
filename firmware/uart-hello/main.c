/* Writes directly to the UART's memory-mapped registers, bypassing syscalls. */

#define UART_BASE 0x10000000
#define UART_THR  (*(volatile char *)UART_BASE)       /* TX register */
#define UART_LSR  (*(volatile char *)(UART_BASE + 5)) /* Line Status */
#define UART_LSR_THR_EMPTY 0x20

static void uart_putchar(char c) {
    while (!(UART_LSR & UART_LSR_THR_EMPTY))
        ;  /* wait until TX ready */
    UART_THR = c;
}

static void uart_puts(const char *s) {
    while (*s)
        uart_putchar(*s++);
    uart_putchar('\n');
}

int main(void) {
    uart_puts("Hello from UART!");
    return 0;
}
