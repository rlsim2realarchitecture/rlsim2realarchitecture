import numpy as np
import pybullet as p

remove_user_item_indices = []
remove_body_indices = []


def draw(pos,
         q_xyzw,
         line_width=4,
         line_length=0.3,
         parent_object_index=-1,
         parent_link_index=0,
         color=[1, 1, 1, 1],
         radius=0.03,
         text=''):
    global remove_user_item_indices
    global remove_body_indices
    create_pose_marker(pos,
                       q_xyzw,
                       text=text,
                       lineWidth=line_width,
                       lineLength=line_length,
                       parentObjectUniqueId=parent_object_index,
                       parentLinkIndex=parent_link_index)


def flush():
    global remove_user_item_indices
    global remove_body_indices
    for idx in remove_user_item_indices:
        p.removeUserDebugItem(idx)
    for idx in remove_body_indices:
        p.removeBody(idx)
    remove_user_item_indices = []


def create_pose_marker(position=np.array([0, 0, 0]),
                       orientation=np.array([0, 0, 0, 1]),
                       text='',
                       xColor=np.array([1, 0, 0]),
                       yColor=np.array([0, 1, 0]),
                       zColor=np.array([0, 0, 1]),
                       textColor=np.array([0, 0, 0]),
                       lineLength=0.1,
                       lineWidth=1,
                       textSize=1,
                       textPosition=np.array([0, 0, 0.1]),
                       textOrientation=None,
                       lifeTime=0,
                       parentObjectUniqueId=-1,
                       parentLinkIndex=-1):
    """Create a pose marker

    Create a pose marker that identifies a position and orientation in space
    with 3 colored lines.

    """
    global remove_user_item_indices
    pts = np.array([[0, 0, 0], [lineLength, 0, 0], [
                   0, lineLength, 0], [0, 0, lineLength]])
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1, :], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity)
    idx = p.addUserDebugLine(po, px, xColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    idx = p.addUserDebugLine(po, py, yColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    idx = p.addUserDebugLine(po, pz, zColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    if textOrientation is None:
        textOrientation = orientation
    idx = p.addUserDebugText(text, [0, 0, 0.1], textColorRGB=textColor,
                             textSize=textSize,
                             parentObjectUniqueId=parentObjectUniqueId,
                             parentLinkIndex=parentLinkIndex)
    remove_user_item_indices.append(idx)
