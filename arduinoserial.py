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