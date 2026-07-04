from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'projet'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['readme.txt']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dev Kumar',
    maintainer_email='devk79036@gmail.com',
    description='ENSAM ROS2 Project - TurtleBot3 navigation challenges',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'corridor_node    = projet.corridor_node:main',
            'line_follow_node = projet.line_follow_node:main',
            'ball_push_node   = projet.ball_push_node:main',
            'full_run_node    = projet.full_run_node:main',
            'main_node        = projet.main_node:main',
        ],
    },
)
