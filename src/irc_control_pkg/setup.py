from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'irc_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pc',
    maintainer_email='pc@pc.com',
    description='IRC 6-DoF robot arm point planning and hardware control package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_motion_control_node = irc_control_pkg.motor_motion_control_node:main',
            'main_irc_node = irc_control_pkg.main_irc:main',
            'point_pose_node = irc_control_pkg.point_pose_node:main',
            'detect_yolo_sam_obb_pub = irc_control_pkg.detect_yolo_sam_obb_pub:main',
            'transform_node = irc_control_pkg.transform_node:main',
        ],
    },
)
