import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro


def generate_launch_description():

    pkg_share = get_package_share_directory('ece_project_description')
    
    controller_config_path = os.path.join(
        pkg_share, 'config', 'ece_project_controllers.yaml'
    )
    
    xacro_file_path = os.path.join(
        pkg_share, 'urdf', 'ece_project.xacro'
    )

    robot_description_config = xacro.process_file(
        xacro_file_path,
        mappings={'controller_config_path': controller_config_path}
    )
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_urdf},
            {'use_sim_time': True}
        ]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ),
        launch_arguments={
            'gz_args': '-r empty.sdf',
            'use_sim_time': 'true'
        }.items()
    )

    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description', 
            '-name', 'ece_project',
            '-z', '0.5'
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster', 
            '--controller-manager', '/controller_manager'
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_config_path
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        gazebo,
        spawn_node,
        bridge_node,
        joint_state_broadcaster_spawner,
        joint_trajectory_controller_spawner,
    ])
