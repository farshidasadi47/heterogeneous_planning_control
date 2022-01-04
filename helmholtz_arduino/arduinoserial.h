/*
  This header file contains class and methods for our serial
  communication protocol.
  The protocols should match those in "arduinoserial.py" module.

  The protocol is for sending array of numerics. Each packet of data
  consists of "4byte delimiter + 1byte size of bytes + serialized data".
  The data serialization is simple cast to float32.
  Maximum integer range that can be sent this way is:
           +/-2^24 = +/- 16777216
  
  Author: Farshid Asadi, farshidasadi47@yahoo.com
  Decemer 2021.
*/
#ifndef ARDUINOSERIAL_H
#define ARDUINOSERIAL_H

#include "Arduino.h"
#include "HardwareSerial.h"
/********* Type defs and globals **************************************/
#undef SERIAL_RX_BUFFER_SIZE
#define SERIAL_RX_BUFFER_SIZE 256
extern HardwareSerial Serial;
/********* Function declarations **************************************/
// some_func
// Description.
// Input: None
// Output: None
template<class T, int size>
class Buffer{
    /* This class provides a special circular buffer that always
       puts new data into the buffer.
       Also for reading purpose it reads the total buffer starting from
       the most recent value.
       The purpose of this buffer is to store last N byte of the serial
       stream, so that we can compare it with our delimiter.
    */
    public:
    // Instance vars
    T fifo[size]{0};
    int putndx{0};    // Index of where to put next.
    int getndx{0};    // Index of where to get next.
    // Constructor and deconstructor
    Buffer(){} 
    ~Buffer(){}
    // Methods
    int put_fifo (T data) {
        int ndx{putndx};                  // Temporary index.
        //ndx = putndx;                   // Copy of put index.
        fifo[ndx] = data;                 // Put data into fifo.
        putndx = (ndx + 1)%size;          // Wrap put index.
    }

    void read_fifo(T* arr){
        // Reads the whole fifo.
        // Input: array of the same size of "size".
        int ndx{putndx};                  // Temporary index.
        // Copy the fifo to array.
        for(int i=0; i<size; i++){
            arr[i] = fifo[(i+ndx)%size];
        }
        return;
    }

    int wrap_int(int x, int const lbx = 0, int const ubx = size){
        // Wraps a given integer between lower and upper bound.
        // Input: integer to be wrapped and bounds.
        // Output: wrapped value.
        int range_size = ubx - lbx + 1;
        if(x < lbx){
            x += range_size * ((lbx - x) / range_size + 1);
        }
        return lbx + (x - ubx) % range_size;
    }

};

Buffer<unsigned char, 4> buffer;

class Arduino{
    /* Establishes serial connection with a simple protocol.
       Messages are numeric arrays formatted as delimiter + size + data.
       The delimiter is 4 bytes: "0xff 0xff 0xff 0x7f"
       The size is one byte shows how many bytes to read.
       The data is a formatted numeric array. Every type is converted to 
       32bit float and then serialized.
       Maximum integer range that can be sent this way is:
           +/-2^24 = +/- 16777216
    */
    
    public:
    // Instance variables.
    unsigned char delimiter_[4]{0xff,0xff,0xff,0x7f};//
    unsigned char read_buffer_[256]{0};
    unsigned long int baud_;
    HardwareSerial *io;
    // Constructor and deconstructor
    Arduino()                      : baud_(115200), io(&Serial){}
    Arduino(unsigned long int baud): baud_(baud)  , io(&Serial){}
    ~Arduino(){}
    // Methods
    void begin(){
       // Starts serial communication.
       io->begin(baud_);
    }
    void end(){
        // Finishes serial communication.
        io->end();
    }

    template<class T>
    void write(T *arr, unsigned char arr_size){
        // This function writes a given array.
        // Length of arr (arr_size) should not exceed 60.
        // Input: Numeric array, size of array.
        // Output: None.
        unsigned char written[4 + 1 + 4*arr_size]{0};
        float serialized[arr_size]{0};
        // Add delimiter to the written array.
        memcpy(&written[0], &delimiter_[0], sizeof(delimiter_));
        // Add size of the message.
        written[sizeof(delimiter_)] = arr_size*sizeof(arr[0]);
        // Convert data to the protocol format explained on top of file.
        Arduino::serialize<T>(arr, serialized, arr_size);
        // Append serialized data to the message.
        memcpy(&written[sizeof(delimiter_) + 1],
               serialized,
               sizeof(serialized));

        // for testing
        Arduino::print_str(written,4 + 1 + 4*arr_size);
        Arduino::print_hex(written,4 + 1 + 4*arr_size);

        return;
    }

    template<class T>
    void serialize(T* arr, float* serialized, unsigned char arr_size){
        // Serializes the data based on out protocol.
        // Input: pointer to the data array.
        // Output: pointer to data array.
        // Multiply the data by 1000, round it, store it in 32bit int.
        for(int i = 0; i< arr_size; i++){
            serialized[i] = (float) arr[i];
        }
        return;
    }

    void print_str(unsigned char* arr, unsigned char arr_size){
        // Prints hex representation of given char array.
        // Input:  Pointer to array to print.
        //         Array size.
        // Output: None.
        char str[arr_size*4+1]{0};
        for(unsigned int j = 0; j < arr_size; j++){
            sprintf(&str[4*j], "%4.c", arr[j]);
        }
        io->println(str);
        return;
    }

    void print_hex(unsigned char* arr, unsigned char arr_size){
        // Prints hex representation of given char array.
        // Input:  Pointer to array to print.
        //         Array size.
        // Output: None.
        char str[arr_size*4+1]{0};
        for(unsigned int j = 0; j < arr_size; j++){
            sprintf(&str[4*j], "\\x%02x", arr[j]);
        }
        io->println(str);
        return;
    }

    void read(){
        // Reads data from serial and deserializes it and puts it in 
        // read_buffer_ variable.
        // Input: None.
        // Output: None.
        unsigned char ndx = 0;
        while (io->available()){
            read_buffer_[ndx] = io->read();
        }

    }

};



#endif // ARDUINOSERIAL_H
