########################################################################
# This is lunch file for camera related nodes.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from launch import LaunchDescription
from launch_ros.actions import Node
########## Body ########################################################
def generate_launch_description():
    ld = LaunchDescription()
    getvideo_node = Node(
        package="swarm",
        executable="getvideo",
        output= "screen",
    )
    showvideo_node = Node(
        package="swarm",
        executable="showvideo"
    )
    ld.add_action(getvideo_node)
    ld.add_action(showvideo_node)
    return ld
