build:
	arm-none-eabi-gcc \
	-mcpu=cortex-m3 \
	-mthumb \
	-nostartfiles \
	-DSTM32F103x6 \
	-Tlinker.ld \
	-I/Users/drbh/Projects/tinyarmc/CMSIS/Include \
	-I/Users/drbh/Projects/cmsis-header-stm32/stm32f1xx/Include \
	-Wl,-Map=output.map \
	-o main.elf \
	startup.s \
	main.c \
	system_stm32f1xx.c \
	mnist_weights.h \
	-Os

strip:
	arm-none-eabi-strip main.elf

bin:
	arm-none-eabi-objcopy -O binary main.elf main.bin

flash:
	st-flash --reset write main.bin 0x08000000

complete: build bin flash

log:
	picocom -b 9600 /dev/tty.usbserial-0001