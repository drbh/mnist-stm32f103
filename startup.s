.section .vectors, "a", %progbits
.word _stack_top        /* Initial stack pointer */
.word Reset_Handler     /* Reset handler */

/* Reset handler implementation */
.section .text.Reset_Handler, "ax", %progbits
.global Reset_Handler
.type Reset_Handler, %function

Reset_Handler:
    ldr r0, =_stack_top  /* Load stack pointer address into r0 */
    mov sp, r0           /* Set stack pointer */
    bl main              /* Call main function */
    b .                  /* Infinite loop if main returns */

/* Default handlers (weak definitions to avoid linker errors) */
.weak NMI_Handler
NMI_Handler:
    b .

.weak HardFault_Handler
HardFault_Handler:
    b .
