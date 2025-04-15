Usage:

terminal 1: py 3dvfh_star_simu.py
terminal 2: py oak_disp_body_map_tf_pub.py
terminal 3: py uav_trajectory_pub.py
terminal 4: rviz2 -d vfh.rviz


turn on disparity subpixel to match data published for VIO's feature matching.

The oak disparity publisher has an option to setup the published image with (data format mono16 uint16) or without (data format mono8 uint8)subpixel turned on.


Make sure to set the proper frame_id if not transformation (tf) is provided (set to body or check the code if changed) and set the QOS to match (to be done in rviz2 dropdown menus)
