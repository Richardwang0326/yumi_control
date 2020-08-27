#!/usr/bin/python

import rospy
import numpy as np
import cv2  # OpenCV module
import time
import message_filters
import math
from sensor_msgs.msg import Image, CameraInfo
import tf
import tf.transformations as tfm
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest,GetPose, GetPoseResponse, \
    YumipyTrigger, YumipyTriggerRequest,MoveGripper, MoveGripperRequest


def Finddepth(depth_data, point):
    
    xp, yp = point[0], point[1]
    # Get the camera calibration parameter for the rectified image
    msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=None)
    #     [fx'  0  cx' Tx]
    # P = [ 0  fy' cy' Ty]
    #     [ 0   0   1   0]

    fx = msg.P[0]
    fy = msg.P[5]
    cx = msg.P[2]
    cy = msg.P[6]

    try:
        cv_depthimage = cv_bridge.imgmsg_to_cv2(depth_data, "32FC1")
        cv_depthimage2 = np.array(cv_depthimage, dtype=np.float32)
    except CvBridgeError as e:
        print(e)

    if not math.isnan(cv_depthimage2[int(yp)][int(xp)]) :
        zc = cv_depthimage2[int(yp)][int(xp)]
        X1 = getXYZ(xp, yp, zc, fx, fy, cx, cy)

    return X1

def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    #### Definition:
    # cx, cy : image center(pixel)fd
    # fx, fy : focal length
    # xp, yp: index of the depth image
    # zc: depth
    inv_fx = 1.0/fx
    inv_fy = 1.0/fy
    x = (xp-cx) *  zc * inv_fx / 1000
    y = (yp-cy) *  zc * inv_fy / 1000
    z = zc / 1000
    return (x,y,z)


def Imageprocess(msg):

    # =====================================================
    # transform cv format and find red region

    cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    blurred_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([4, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask_red

    result = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    # =================================================
    
    # ======================================================================================================================================
    # detect contour 
    
    gray = cv2.cvtColor(result ,cv2.COLOR_BGR2GRAY)

    im, contours, hierarchy = cv2.findContours(
        gray,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(cv_image,contours,-1,(0,0,255),3)
    output_result.publish(cv_bridge.cv2_to_imgmsg(cv_image, encoding="passthrough"))
    # ====================================================================================================================================
    
    # ============================================
    # find point and quaternion

    for c in contours:
	    rect = cv2.minAreaRect(c)

    if rect[1][0] > rect[1][1]:
    	angle=-90 - rect[2]
    else:
        angle=-rect[2]

    q = tf.transformations.quaternion_from_euler(0, 3.1415, 1.57-angle*3.1415/180, axes="rxyz")

    return rect[0], q
    # ============================================

def transform_pose_to_base_link(t,q):

    euler_pose = tfm.euler_from_quaternion(q)
    tf_cam_col_opt_fram = tfm.compose_matrix(translate=t, angles=euler_pose)
    
    trans, quat = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))

    euler = tfm.euler_from_quaternion(quat)
    tf = tfm.compose_matrix(translate=trans, angles=euler)
    
    t_pose = np.dot(tf, tf_cam_col_opt_fram)

    t_ba_li = t_pose[0:3,3]

    return t_ba_li

def summer_arm_control_cb(req):

    rospy.loginfo("summer_arm_control_called")
    color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
    point, q= Imageprocess(color_msg)
    depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
    real_world_point = Finddepth(depth_msg, point)

    #quaternion = [0.004, 0.003, -0.923, -0.386]  # w x y z (please fix your quaternion)
    position = transform_pose_to_base_link(real_world_point, q)

    meter_to_mm = 1000
    req = GotoPoseRequest()
    req.wait_for_res = True

    if position[1] >= 0 :
        req.arm = 'left'
    else:
        req.arm = 'right'

    req.quat = [q[3],q[0],q[1],q[2]]

    req.position = [position[0]*meter_to_mm, position[1]*meter_to_mm, position[2]*meter_to_mm+50]
    go_pose_plan_srv(req)
    rospy.sleep(1)

    move_gripper_srv(MoveGripperRequest(arm=req.arm, width=0.025))
    rospy.sleep(1)

    req.position = [position[0]*meter_to_mm, position[1]*meter_to_mm, position[2]*meter_to_mm+20]
    go_pose_plan_srv(req)
    rospy.sleep(1)

    move_gripper_srv(MoveGripperRequest(arm=req.arm, width=0))
    rospy.sleep(1)

    req.position = [position[0]*meter_to_mm, position[1]*meter_to_mm, position[2]*meter_to_mm+50]
    go_pose_plan_srv(req)
    rospy.sleep(1)

    goto_wait_joint_srv(req.arm)

    return TriggerResponse()

if __name__=="__main__":

    rospy.init_node("detection", anonymous=True)
    listener = tf.TransformListener()

    # Bridge to convert ROS Image type to OpenCV Image type
    cv_bridge = CvBridge()

    go_pose_plan_srv = rospy.ServiceProxy("/goto_pose_plan", GotoPose)
    move_gripper_srv = rospy.ServiceProxy("/move_gripper", MoveGripper)
    goto_wait_joint_srv = rospy.ServiceProxy("/goto_wait_joint", YumipyTrigger)
    rospy.Service("~/summer_arm_control", Trigger, summer_arm_control_cb)
    output_result = rospy.Publisher('/camera/color/image_result', Image, queue_size=10)

    rospy.spin()
