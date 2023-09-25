from setuptools import setup
import os.path

package_name = 'detectron_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
	data_files=[
		('share/ament_index/resource_index/packages',
			['resources/' + package_name]),
		('share/' + package_name, ['package.xml']),
		(os.path.join('share', package_name, 'resources'), ['resources/Untitled.png']),
	],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pepe',
    maintainer_email='ojedamorala@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detectron_ros_node = detectron_ros.detectron_ros:main',
            'test = detectron_ros.test:main',
        ],
    },
)
