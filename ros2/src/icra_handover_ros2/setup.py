from glob import glob
from setuptools import find_packages, setup

package_name = "icra_handover_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ICRA Vision Module Maintainers",
    maintainer_email="todo@example.com",
    description="ROS2 adapter nodes for the ICRA vision handover Python runtime.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dual_camera_node = icra_handover_ros2.dual_camera_node:main",
            "perception_node = icra_handover_ros2.perception_node:main",
            "robot_node = icra_handover_ros2.robot_node:main",
            "handover_coordinator_node = icra_handover_ros2.handover_coordinator_node:main",
            "visualization_node = icra_handover_ros2.visualization_node:main",
        ],
    },
)
