########################################################################
# This files hold classes and functions that controls swarm of milirobot 
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import time
import struct
from array import array

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
        self.__buffer_size = 255
        # Communication protocol related vars
        self.__define_protocol_vars()
        # Timing related vars/
        self.__last_spin_time = None
        self.__hz = hz
        self.__period = 1/hz
        self.__period_ns = int(round(1000000000/hz,0))

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
            raise IOError("No Arduino found. Connect your Board.")
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
        self.__delimiter = b'\x7f\xff\xff\xff'
        self.__format = 'little'
        # Data types available in the protocol with their sizes.
        self.__data_types = {}
        self.__data_types['int32'] = 4
        # Following ranges are based on the chosen delimiter.
        self.__max_int32 = struct.unpack('!i',b'\x7f\xff\xff\xff')[0]
        self.__min_int32 = struct.unpack('!i',b'\x80\x00\x00\x00')[0]
    
    def begin(self):
        """Establishes a serial connection with the chosen port
        at the given baud rate."""
        try:
            self.connection = serial.Serial(self.__port,self.__baud)
            # Set buffer sizes, this is a suggestion to the hardware
            # driver, and may or may not over-write the driver's value.
            self.connection.set_buffer_size(rx_size = self.__buffer_size,
                                            tx_size = self.__buffer_size)
            time.sleep(.1)
            print(f"Connection to {self.__port}.")
            self.__last_spin_time = time.perf_counter_ns()
        except serial.serialutil.SerialException:
            print(line_sep)
            print(f"Serial port {self.__port} is busy.")
            print("Close the corresponding process and restart the program.")
    
    def close(self):
        """Closes the connection."""
        if self.connection is not None:
            self.connection.close()
            print(f"Disconnected from {self.__port}.")

########## test section ################################################
if __name__ == '__main__':

    with Arduino() as arduino:
        arduino.begin()
        #while True:
        #    print("{:+011.3f}".format(time.perf_counter()))
        #    arduino.sleep()