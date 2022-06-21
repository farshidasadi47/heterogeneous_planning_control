from setuptools import setup
from setuptools import find_packages

package_name = 'swarm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='farshid asadi',
    maintainer_email='farshidasadi47@yahoo.com',
    description='Swarm of magnetic milli-robot planning and control',
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
