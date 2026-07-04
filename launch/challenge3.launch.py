#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    challenge_pkg_dir = get_package_share_directory('challenge_project')
    projet_pkg_dir    = get_package_share_directory('projet')

    x_arg   = DeclareLaunchArgument('x_pose',    default_value='-4.5')
    y_arg   = DeclareLaunchArgument('y_pose',    default_value='0.0')
    yaw_arg = DeclareLaunchArgument('yaw_angle', default_value='3.14159')

    set_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger')

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(challenge_pkg_dir, 'launch', 'projet.launch.py')),
        launch_arguments={
            'x_pose':    LaunchConfiguration('x_pose'),
            'y_pose':    LaunchConfiguration('y_pose'),
            'yaw_angle': LaunchConfiguration('yaw_angle'),
        }.items(),
    )

    ball_node = Node(
        package='projet',
        executable='ball_push_node',
        name='ball_pusher',
        output='screen',
        parameters=[os.path.join(projet_pkg_dir, 'config', 'params.yaml')],
    )

    return LaunchDescription([set_model, x_arg, y_arg, yaw_arg, simulation, ball_node])
