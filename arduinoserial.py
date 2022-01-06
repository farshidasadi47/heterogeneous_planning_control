########################################################################
# This files hold classes and functions that controls swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import time
import re
import struct
from array import array

from ctypes import windll #new
timeBeginPeriod = windll.winmm.timeBeginPeriod #new
timeBeginPeriod(1) #new

import serial
import serial.tools.list_ports
import numpy as np

np.set_printoptions(precision=2, suppress=True)
line_sep = "#"*79
########## classes and functions #######################################
class Arduino():
    """
    This class manages communication with arduino through strings.
    """
    def __init__(self, hz = 10,baud = 115200):
        # Serial configuration vars.
        self.__baud = baud
        self.__port = None
        self.__find_arduino()  # Find and set the com port.
        self.connection = None
        self.__buffer_size = 65
        # Communication protocol related vars
        self.__define_protocol_vars()
        # Timing related vars/
        self.__last_spin_time = None
        self.__hz = hz
        self.__period = 1/hz
        self.__period_ns = int(round(1000000000/hz,0))
        #
        self.written = None  # Last written bytes.
        self.remnant = b''

    def __enter__(self):
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        self.close()

    def __find_arduino(self):
        """
        This method searches for available Arduino boards in
        initialization, then prompts available ones and wants user
        to choose the currect one.
        """
        # Get list of the available serial ports.
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        # Check if there is any Arduino port.
        for port in ports:
            if "Arduino" in port.description:
                arduino_ports.append(port)
        # Raise an error if there was no arduino connected.
        if not arduino_ports:
            raise IOError
            print("No Arduino found. Connect your Board.")
        # If there was arduinos, choose the board.
        if len(arduino_ports)<1:
            # If there is only one arduino choose that and proceed.
            port = arduino_ports[0]
            print(f"Arduino {port.description.split(' ')[1]}"
                  f" at port {port.device} is selected.")
        else:
            # If there are multiple boards, print list of the boards.
            # Then let user choose the appropriate board.
            print(line_sep)
            print("List of available boards:")
            for index, port in enumerate(arduino_ports):
                print(f"{index+1}- Arduino {port.description.split(' ')[1]}"
                      f" at port {port.device}.")
            # Ask the user to choose the board.
            while True:
                # Keep asking while appropriate number is enterred.
                try:
                    index = int(input("Please enter the board number: ")) - 1

                    # Check if the number is in the available range.
                    if index not in range(len(arduino_ports)):
                        print("Oops!  That was out of range.  Try again...")
                        continue
                    else:
                        port = arduino_ports[index]
                        break

                except ValueError:
                    # If user input is not enteger show error.
                    # and reask for board selection.
                    print("Oops!  That was no valid number.  Try again...")
        # Save the selected board port name. 
        self.__port = port.device

    def __define_protocol_vars(self):
        """
        Defines variables related to our messaging protocol.
        """
        # Everything is little endian, since ATmega328P is this way.
        self.__delimiter = b'\x7f\xff\xff\xff'[::-1]
        self.__format = 'little'
        # Data types available in the protocol with their sizes.
        self.__type_size = {}
        self.__type_size['float32'] = 4
        # Following ranges are based on the chosen delimiter.
        self.__max_float32 = struct.unpack('!i',b'\x7f\xff\xff\xff')[0]
        self.__min_float32 = struct.unpack('!i',b'\x80\x00\x00\x00')[0]
    
    def sleep(self):
        """
        Sleeps based on the specified rate of the class.
        """
        ctime = time.perf_counter()
        elapsed = ctime - self.__last_spin_time
        sleep_value = max(self.__period - elapsed, 0 )
        time.sleep(sleep_value)
        self.__last_spin_time = time.perf_counter()
        

    def begin(self):
        """Establishes a serial connection with the chosen port
        at the given baud rate."""
        try:
            self.connection = serial.Serial(self.__port,self.__baud,
                                            timeout=0, write_timeout=1e-3)
            # Set buffer sizes, this is a suggestion to the hardware
            # driver, and may or may not over-write the driver's value.
            self.connection.set_buffer_size(rx_size = self.__buffer_size,
                                            tx_size = self.__buffer_size)
            time.sleep(.1)
            print(f"Connection to {self.__port}.")
            self.__last_spin_time = time.perf_counter()
        except serial.serialutil.SerialException:
            print(line_sep)
            print(f"Serial port {self.__port} is busy.")
            print("Close the corresponding process and restart the program.")
    
    def close(self):
        """Closes the connection."""
        if self.connection is not None:
            self.connection.close()
            print(f"Disconnected from {self.__port}.")

    def write(self, x):
        """
        This function will sends a given list of numbers.
        It convrts everynumber to 32bit float and sends the delimited
        byte array. The message format is as:
            delimiter + size of array in bytes + message
        The array length should not exceed 60 members.
        Max integer range to be sent accurately is 2^24 = 
        """
        # Get size of the input.
        size = len(x)
        # Convert to numpy array for easier processing.
        if type(x) is not np.ndarray:
            x = np.array(x).squeeze()
        # Convert to signed int32 binary in little endian format.
        x = struct.pack(f'<{size}f',*x)
        # Add delimiter and size to the beginning of the message.
        self.written = self.__delimiter + struct.pack('B',4*size) + x
        # Write the message.
        #self.connection.write(self.written)
        return self.written
    
    def read(self):
        """Reads serial buffer and returns the latest complete array."""
        new_data = False
        # Read the whole available buffer
        buffer = self.connection.read(4096)
        # Add the remnant from previous buffer read to current one.
        # This ensures that newest message is not lost.
        sent = self.remnant + buffer
        # Split based on delimiter, includes the delimiter in the list.
        splitted = re.split(b'('+self.__delimiter+b')',sent)
        # Extract the message.
        """ if len(splitted)>1:
            # There was at least one occurence of the delimiter.
            temp = splitted[-1]
            size = temp[0]//self.__type_size['float32']  # array size.
            # If the last piece is complete, return it.
            if len(temp) == (temp[0]+1):
                data = struct.unpack(f'<{size}f')
                self.__type_size['float32'] """




        return buffer

    

def main():
    """This function isolates tests from the module."""
    with Arduino() as arduino:
        #arduino.begin()
        #while True:
        #    arduino.sleep()
        pass
########## test section ################################################
if __name__ == '__main__':
    reads = []
    with Arduino(10) as arduino:
        arduino.begin()
        time.sleep(5)
        counter = 0
        start = time.time()
        while counter<50:
            x = arduino.read()
            if x:
                print(x)
                reads.append(x)
            #print(f"{time.perf_counter():+011.4f}")
            counter += 1
            arduino.sleep()
        end = time.time()
        print(f"Total time for counter {counter:04d} loops: {end-start:+09.4f}secs")
        print(f"Time per loop: {(end-start)/counter*1000:+015.4f} msecs")
        
        
# %%
