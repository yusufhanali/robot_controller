from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kovan',
    maintainer_email='akgulburak01@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = robot_controller.new_controller:main',
            'head_detector = robot_controller.head_detector:main',
            'gaze_publisher = robot_controller.gaze_publisher:main',
            'gaze_visualizer = robot_controller.gaze_visualizer:main',
            'gaze_test_data_collecter = robot_controller.gaze_test_data_collecter:main',
        ],
    },
)
