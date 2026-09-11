from setuptools import find_packages, setup

package_name = "agentic_grasp_stack_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dheeraj Kiran",
    maintainer_email="denna@asu.edu",
    description=(
        "Publishes Mac-side Grounding DINO perception detections "
        "(JSON hand-off file) onto a ROS2 topic."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detection_publisher = agentic_grasp_stack_perception.detection_publisher_node:main",
        ],
    },
)
