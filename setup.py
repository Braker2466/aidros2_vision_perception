from setuptools import find_packages, setup

package_name = 'vision_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/vision_perception', ['resource/cutoff_yolov5s_sigmoid_w8a8.qnn229.ctx.bin','resource/test1.jpg','resource/testvideo1.mp4','resource/testvideo2.mp4'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aidlux',
    maintainer_email='aidlux@todo.todo',
    description='yolo&face',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_recognition_node = vision_perception.face_recognition_node:main',
            'yolo_detector_node = vision_perception.yolo_detector_node:main',
            'video_publisher = vision_perception.video_publisher:main',
            'cam_pub = vision_perception.cam_pub:main',
        ],
    },
)
