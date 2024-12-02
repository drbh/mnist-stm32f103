# MNIST from scratch on STM32F103 (Blue Pill)

This project demonstrates how to train a simple model on the MNIST dataset and deploy it to an STM32F103 microcontroller. The model is trained using PyTorch and the weights are saved to a header file that can be included in the embedded project.

## Requirements

- arm-none-eabi-gcc
- st-flash
- uv
- picocom

## Getting Started

```bash
uv venv
uv pip install torch torchvision numpy pyserial matplotlib
```

## Training the model

First we'll train a simple model on the MNIST dataset and save the weights directly into a file

```bash
uv run train-save.py
# Using device: cpu
# Train Epoch: 1 [0/60000 (0%)]	Loss: 2.341760
# Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.860331
# Train Epoch: 1 [12800/60000 (21%)]    Loss: 0.569391
# Train Epoch: 1 [19200/60000 (32%)]    Loss: 0.442725
# Train Epoch: 1 [25600/60000 (43%)]    Loss: 0.387216
# Train Epoch: 1 [32000/60000 (53%)]    Loss: 0.363262
# Train Epoch: 1 [38400/60000 (64%)]    Loss: 0.345553
# Train Epoch: 1 [44800/60000 (75%)]    Loss: 0.319755
# Train Epoch: 1 [51200/60000 (85%)]    Loss: 0.324141
# Train Epoch: 1 [57600/60000 (96%)]    Loss: 0.393049
# Test set: Average loss: 0.3340, Accuracy: 9096/10000 (90.96%)
# ...
# Train Epoch: 10 [0/60000 (0%)]    Loss: 0.171914
# Train Epoch: 10 [6400/60000 (11%)]    Loss: 0.211914
# Train Epoch: 10 [12800/60000 (21%)]   Loss: 0.360202
# Train Epoch: 10 [19200/60000 (32%)]   Loss: 0.576416
# Train Epoch: 10 [25600/60000 (43%)]   Loss: 0.218224
# Train Epoch: 10 [32000/60000 (53%)]   Loss: 0.171839
# Train Epoch: 10 [38400/60000 (64%)]   Loss: 0.358727
# Train Epoch: 10 [44800/60000 (75%)]   Loss: 0.458519
# Train Epoch: 10 [51200/60000 (85%)]   Loss: 0.215831
# Train Epoch: 10 [57600/60000 (96%)]   Loss: 0.361634
# Test set: Average loss: 0.2640, Accuracy: 9264/10000 (92.64%)
#
#
# Best model accuracy: 92.82%
#
# Weights saved to mnist_weights.h
```

## Deploying the model

now connect our programmer to the board to flash the model

| ST-Link V2 | STM32F103 |
| ---------- | --------- |
| GND        | GND       |
| SWCLK      | SWCLK     |
| SWDIO      | SWDIO     |
| 3.3V       | 3.3V      |

build for arm-none-eabi-gcc and flash the board

```bash
make complete
# arm-none-eabi-gcc \
#   -mcpu=cortex-m3 \
#   -mthumb \
#   -nostartfiles \
#   -DSTM32F103x6 \
#   -Tlinker.ld \
#   -I/Users/drbh/Projects/tinyarmc/CMSIS/Include \
#   -I/Users/drbh/Projects/cmsis-header-stm32/stm32f1xx/Include \
#   -Wl,-Map=output.map \
#   -o main.elf \
#   startup.s \
#   main.c \
#   system_stm32f1xx.c \
#   mnist_weights.h \
#   -Os
# arm-none-eabi-objcopy -O binary main.elf main.bin
# st-flash --reset write main.bin 0x08000000
# st-flash 1.8.0
# 2024-12-02T17:24:45 INFO common.c: STM32F1xx_MD: 20 KiB SRAM, 64 KiB flash in at least 1 KiB pages.
# file main.bin md5 checksum: af9db4f9385d7a7bc385d948bcdb6bda, stlink checksum: 0x0043190c
# 2024-12-02T17:24:45 INFO common_flash.c: Attempting to write 34700 (0x878c) bytes to stm32 address: 134217728 (0x8000000)
# -> Flash page at 0x8008400 erased (size: 0x400)
# 2024-12-02T17:24:46 INFO flash_loader.c: Starting Flash write for VL/F0/F3/F1_XL
# 2024-12-02T17:24:46 INFO flash_loader.c: Successfully loaded flash loader in sram
# 2024-12-02T17:24:46 INFO flash_loader.c: Clear DFSR
#  34/34  pages written
# 2024-12-02T17:24:48 INFO common_flash.c: Starting verification of write complete
# 2024-12-02T17:24:48 INFO common_flash.c: Flash written and verified! jolly good!
# 2024-12-02T17:24:48 INFO common.c: NRST is not connected --> using software reset via AIRCR
# 2024-12-02T17:24:48 INFO common.c: Go to Thumb mode
```

now we can disconnect the programmer and connect the serial to USB to see the output. The model will take a 28x28 image of a digit and output the predicted digit.

| Serial To USB | STM32F103 |
| ------------- | --------- |
| GND           | GND       |
| TX            | A10       |
| RX            | A9        |
| 3.3V          | 3.3V      |


## Running inference

finally we can run the inference script to send an image to the board and see the prediction.

this will open an example of the image on your computer, and once you close it, it will send the image to the board and print the prediction.

The predicted output from the embedded model will be printed `Predicted class: X` and the logits for each class will be printed as well.

```bash
uv run infer.py
# Connected to /dev/tty.usbserial-0001 at 115200 baud
#
# Sending image (true label: 7)
# 2024-12-02 17:31:23.428 python3[27036:4601661] +[IMKClient subclass]: chose IMKClient_Modern
# 2024-12-02 17:31:23.428 python3[27036:4601661] +[IMKInputSession subclass]: chose IMKInputSession_Modern
# Sending data...
# Sending 784 bytes
# Waiting for prediction...
# Image data received successfully
# Starting inference...
# Predicted class: 7
# Logits:
# Class 0: -9.16
# Class 1: -14.20
# Class 2: 1.72
# Class 3: 0.13
# Class 4: -1.51
# Class 5: -7.06
# Class 6: -7.65
# Class 7: 8.81
# Class 8: -2.22
# Class 9: 1.43
#
# Ready for next image
#
# Press Enter to send next image, 'q' to quit:
```

Feel free to try different images and see how the model performs, or to improve the model and see how it affects the performance on the board. This is a simple example, but it can be a good starting point for more complex models and applications!

### Gotcha

- you may need to change the serial port in the `infer.py` script to match the one on your computer (you can find it using `ls /dev/tty.*`)