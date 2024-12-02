/* Linker script for STM32F1xx microcontroller */
OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

/* Memory region definitions */
MEMORY
{
    flash (rx)  : ORIGIN = 0x08000000, LENGTH = 64K  /* 64KB Flash */
    ram   (rwx) : ORIGIN = 0x20000000, LENGTH = 20K  /* 20KB RAM */
}

/* Symbol definitions */
PROVIDE(_stack_top = ORIGIN(ram) + LENGTH(ram));
PROVIDE(_heap_start = _end);
PROVIDE(_heap_end = _stack_top - 0x400); /* Reserve 1KB for stack */

/* Entry point */
ENTRY(Reset_Handler)

SECTIONS
{
    /* Vector table and reset handler */
    .vector :
    {
        . = ALIGN(4);
        _svector = .;
        KEEP(*(.vectors))         /* Keep vector table */
        . = ALIGN(4);
        _evector = .;
    } > flash
    
    /* Code and read-only data */
    .text :
    {
        . = ALIGN(4);
        _stext = .;
        *(.text*)                 /* Code */
        *(.rodata*)               /* Read-only data */
        . = ALIGN(4);
        _etext = .;
    } > flash

    /* Data initialized in Flash, loaded into RAM */
    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > ram AT > flash

    /* Zero-initialized data */
    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)                 /* Common symbols */
        . = ALIGN(4);
        _ebss = .;
    } > ram

    /* End of used RAM */
    . = ALIGN(4);
    _end = .;

    /* Discard unused sections */
    /DISCARD/ :
    {
        *(.comment)               /* Debug comments */
        *(.note.*)                /* Notes */
    }
}

/* Provide initialization sizes */
PROVIDE(_data_size = SIZEOF(.data));
PROVIDE(_data_flash_start = LOADADDR(.data));
PROVIDE(_bss_size = SIZEOF(.bss));
