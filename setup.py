from setuptools import setup
from setuptools import find_packages

import os
from glob import glob

package_name = 'swarm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='farshid asadi',
    maintainer_email='farshidasadi47@yahoo.com',
    description='Motion Planing for Multiple Heterogeneous Magnetic Robots with Unified Input',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open = swarm.rosopen:main',
            'closed = swarm.rosclosed:main',
            'getvideo = swarm.rosclosed:get_video',
            'showvideo = swarm.rosclosed:show_video'
        ],
    },
)
