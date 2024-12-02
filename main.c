#include "stm32f103x6.h"
#include "mnist_weights.h" // MNIST image
#include <string.h>        // For strlen()

#define IMAGE_SIZE (28 * 28) // MNIST images are 28x28 pixels
#define NUM_CLASSES 10       // 10 digits (0-9)

#define IMAGE_BUFFER_SIZE 784
#define UART_TIMEOUT 1000000 // Timeout counter for UART reception

void SystemClock_Config(void)
{
    // Enable HSI (Internal RC Oscillator)
    RCC->CR |= RCC_CR_HSION; // Turn on HSI
    while (!(RCC->CR & RCC_CR_HSIRDY))
        ; // Wait until HSI is ready

    // Select HSI as system clock
    RCC->CFGR &= ~RCC_CFGR_SW;    // Clear clock switch bits
    RCC->CFGR |= RCC_CFGR_SW_HSI; // Select HSI as system clock
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_HSI)
        ; // Wait until HSI is used

    // Update SystemCoreClock variable
    SystemCoreClock = 8000000; // 8 MHz
}

void USART1_Init(uint32_t baudrate)
{
    // Enable clocks for USART1 and GPIOA
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN; // Enable USART1 clock
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;   // Enable GPIOA clock

    // Configure PA9 (TX) as alternate function push-pull
    GPIOA->CRH &= ~(GPIO_CRH_MODE9 | GPIO_CRH_CNF9);
    GPIOA->CRH |= (GPIO_CRH_MODE9_1 | GPIO_CRH_CNF9_1); // Output 2MHz, AF push-pull

    // Configure PA10 (RX) as input floating
    GPIOA->CRH &= ~(GPIO_CRH_MODE10 | GPIO_CRH_CNF10);
    GPIOA->CRH |= GPIO_CRH_CNF10_0; // Floating input

    // Disable USART for configuration
    USART1->CR1 &= ~USART_CR1_UE;

    // Set baud rate
    uint32_t usartdiv = (8000000 + (baudrate / 2)) / baudrate; // Use 8 MHz clock
    USART1->BRR = usartdiv;

    // Configure frame: 8 data bits, no parity, 1 stop bit
    USART1->CR1 &= ~USART_CR1_M;    // 8 data bits
    USART1->CR1 &= ~USART_CR1_PCE;  // No parity
    USART1->CR2 &= ~USART_CR2_STOP; // 1 stop bit

    // Enable transmitter and receiver
    USART1->CR1 |= USART_CR1_TE | USART_CR1_RE;

    // Enable USART
    USART1->CR1 |= USART_CR1_UE;
}

// USART1 Send Character
void USART1_SendChar(char c)
{
    while (!(USART1->SR & USART_SR_TXE))
        ; // Wait until transmit data register is empty
    USART1->DR = c;
}

// USART1 Send String
void USART1_SendString(const char *str)
{
    while (*str)
    {
        USART1_SendChar(*str++);
    }
}

// USART1 Send Integer
void USART1_SendInt(int num)
{
    if (num == 0)
    {
        USART1_SendChar('0');
        return;
    }

    char buf[10];
    int i = 0;

    if (num < 0)
    {
        USART1_SendChar('-');
        num = -num;
    }

    while (num > 0)
    {
        buf[i++] = (num % 10) + '0';
        num /= 10;
    }

    for (int j = i - 1; j >= 0; j--)
    {
        USART1_SendChar(buf[j]);
    }
}

// Fully connected layer forward pass
void fully_connected_forward(float *output, const float *input,
                             const float weights[NUM_CLASSES][IMAGE_SIZE],
                             const float *bias, int input_size, int output_size)
{
    for (int i = 0; i < output_size; i++)
    {
        output[i] = bias[i];
        for (int j = 0; j < input_size; j++)
        {
            output[i] += input[j] * weights[i][j];
        }
    }
}

// Argmax function to find the index of the maximum value in an array
int argmax(const float *array, int size)
{
    int max_index = 0;
    float max_value = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_value)
        {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}

void delay(volatile uint32_t count)
{
    while (count--)
    {
        __NOP();
    }
}


// Function to receive image data over UART
uint8_t receive_image_data(uint8_t *buffer)
{
    uint16_t timeout_counter = 0;
    uint16_t bytes_received = 0;

    USART1_SendString("Waiting for image data...\r\n");

    while (bytes_received < IMAGE_BUFFER_SIZE)
    {
        // Wait for data with timeout
        while (!(USART1->SR & USART_SR_RXNE))
        {
            timeout_counter++;
            if (timeout_counter >= UART_TIMEOUT)
            {
                USART1_SendString("Timeout waiting for data\r\n");
                return 0; // Return failure
            }
        }

        // Reset timeout counter
        timeout_counter = 0;

        // Read byte and store in buffer
        buffer[bytes_received] = USART1->DR;
        bytes_received++;
    }

    USART1_SendString("Image data received successfully\r\n");
    return 1; // Return success
}

int main(void)
{
    // Initialize system clock
    SystemClock_Config();

    // Initialize USART1 with higher baud rate for faster data transfer
    USART1_Init(115200);

    // Array to hold input image data
    uint8_t image_buffer[IMAGE_BUFFER_SIZE] = {0};

    // Array to hold logits (output of the fully connected layer)
    float logits[NUM_CLASSES] = {0};

    USART1_SendString("Ready to receive image data (784 bytes)\r\n");

    while (1)
    {
        // Receive image data
        if (!receive_image_data(image_buffer))
        {
            continue; // If reception failed, restart loop
        }

        // Convert uint8_t values to the format expected by your inference function
        float normalized_image[IMAGE_BUFFER_SIZE];
        for (int i = 0; i < IMAGE_BUFFER_SIZE; i++)
        {
            normalized_image[i] = image_buffer[i] / 255.0f;
        }

        USART1_SendString("Starting inference...\r\n");

        // Perform inference (assuming your function expects float input)
        fully_connected_forward(logits, normalized_image, fc1_weights, fc1_bias, IMAGE_SIZE, NUM_CLASSES);

        // Find predicted class
        int prediction = argmax(logits, NUM_CLASSES);

        // Send prediction over USART1
        USART1_SendString("Predicted class: ");
        USART1_SendInt(prediction);
        USART1_SendString("\r\n");

        // Send logits for debugging
        USART1_SendString("Logits:\r\n");
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            USART1_SendString("Class ");
            USART1_SendInt(i);
            USART1_SendString(": ");

            // Convert float to string
            int integer_part = (int)logits[i];
            int fractional_part = (int)((logits[i] - integer_part) * 100);

            USART1_SendInt(integer_part);
            USART1_SendChar('.');
            if (fractional_part < 0)
                fractional_part = -fractional_part;
            if (fractional_part < 10)
                USART1_SendChar('0');
            USART1_SendInt(fractional_part);
            USART1_SendString("\r\n");
        }

        // Signal ready for next image
        USART1_SendString("\r\nReady for next image\r\n");
    }

    return 0;
}