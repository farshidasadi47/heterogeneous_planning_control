#%%
########################################################################
# This is lunch file for camera related nodes.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
########## Body ########################################################
def generate_launch_description():
    ld = LaunchDescription()
    
    serial_node = ExecuteProcess(
        cmd=["ros2", "run", "micro_ros_agent", "micro_ros_agent",
             "serial", "--dev", "/dev/ttyACM0"
        ],
    )
    ld.add_entity(serial_node)
    return ld
