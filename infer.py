import serial
import time
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def send_image(serial_port, image_data):
    # convert tensor to numpy if needed
    if torch.is_tensor(image_data):
        image_data = image_data.numpy()

    # ensure the image is flattened and in uint8 format (0-255)
    if image_data.dtype != np.uint8:
        image_data = (image_data * 255).astype(np.uint8)

    flat_data = image_data.flatten()
    assert len(flat_data) == 784, f"Expected 784 values, got {len(flat_data)}"

    # send each byte individually
    image_as_bytes = flat_data.tobytes()
    print(f"Sending {len(image_as_bytes)} bytes")

    # send all the data at once
    serial_port.write(image_as_bytes)
    serial_port.flush()

    # wait for data to be sent
    time.sleep(0.1)


def display_image(image_data, prediction=None):
    """Display the image and its prediction if available"""
    plt.figure(figsize=(4, 4))
    plt.imshow(image_data, cmap="gray")
    if prediction is not None:
        plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()


def main():
    # UART Configuration
    PORT = "/dev/tty.usbserial-0001"
    BAUD_RATE = 115200

    # load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    try:
        # open serial port
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {PORT} at {BAUD_RATE} baud")

        # process images one at a time
        for batch_idx, (data, target) in enumerate(test_loader):
            image = data[0][0]  # get first image from batch (removes channel dimension)
            true_label = target[0].item()

            # display the image we're about to send
            print(f"\nSending image (true label: {true_label})")
            display_image(image)

            # send the image
            print("Sending data...")
            send_image(ser, image)

            # read and display the response
            print("Waiting for prediction...")
            response = []
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode("ascii").strip()
                    print(line)
                    response.append(line)
                    if "Ready for next image" in line:
                        break

            # ask if user wants to continue
            user_input = input("\nPress Enter to send next image, 'q' to quit: ")
            if user_input.lower() == "q":
                break

    except serial.SerialException as e:
        print(f"Error: {e}")
    finally:
        if "ser" in locals():
            ser.close()
            print("Serial port closed")


if __name__ == "__main__":
    main()
