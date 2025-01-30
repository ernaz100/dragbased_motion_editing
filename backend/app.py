import sys
import os

from backend.human3dml_util.mini_prior_mdm import mini_prior_mdm
from backend.human3dml_util.prepare_smpl_for_priormdm import prepare_smpl_for_priorMDM
from diffusion_motion_inbetweening import generate_inbetween_motion

from sequence_network import MotionSequenceNetwork
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from pose_estimator import PoseEstimator
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from pose_network import PoseNetwork
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize pose estimator
pose_estimator = PoseEstimator()
joint_mapping = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 22, # left hand
            19: 14, # right collar
            20: 17, # right shoulder
            21: 19, # right elbow
            22: 21, # right wrist
            23: 23, # right hand
        }
joint_mapping_no_hands = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 14, # right collar
            19: 17, # right shoulder
            20: 19, # right elbow
            21: 21, # right wrist
        }
joint_mapping_no_hands_no_pelvis = {
            0: 0,   # left hip 
            1: 3,   # left knee
            2: 6,   # left ankle
            3: 9,  # left foot
            4: 1,   # right hip
            5: 4,   # right knee
            6: 7,   # right ankle
            7: 10,  # right foot
            8: 2,  # spine1
            9: 5,  # spine2
            10: 8,  # spine3
            11: 11, # neck
            12: 14, # head
            13: 12, # left collar
            14: 15, # left shoulder
            15: 17, # left elbow
            16: 19, # left wrist
            17: 13, # right collar
            18: 16, # right shoulder
            19: 18, # right elbow
            20: 20, # right wrist
        }


# Initialize pose network
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
pose_network = PoseNetwork()
checkpoint = torch.load('checkpoints/model_training_batch_256_epochs25.pth', map_location=device, weights_only=True)
pose_network.load_state_dict(checkpoint['model_state_dict'])
pose_network.to(device)
pose_network.eval()
sequence_network = MotionSequenceNetwork()
seq_checkpoint = torch.load('sequence_checkpoints/best_sequence_model.pth', map_location=device, weights_only=True)
sequence_network.load_state_dict(seq_checkpoint['model_state_dict'])
sequence_network.to(device)
sequence_network.eval()
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/estimate_pose', methods=['POST'])
def handle_pose_estimation():
    try:
        data = request.get_json()
        logger.info("Received pose estimation request")
        
        if 'joint_positions' not in data:
            logger.error("No joint positions provided in request")
            return jsonify({'error': 'No joint positions provided'}), 400
            
        joint_positions = np.array(data['joint_positions'])
        selected_joint = data['selected_joint'] 
        
        logger.info(f"Received joint positions with shape: {joint_positions.shape}")
        logger.info(f"Selected joint: {selected_joint}")
        
        # Validate input shape
        if joint_positions.shape != (24, 3):
            logger.error(f"Invalid joint positions shape: {joint_positions.shape}")
            return jsonify({'error': 'Invalid joint positions format. Expected shape: (24, 3)'}), 400

        # Debug original input
        visualize_debug_joints(joint_positions, "Original Input", "static/debug_1_original.png")
        logger.info(f"Original joints stats - min: {joint_positions.min()}, max: {joint_positions.max()}")
        
        # Remap joints to SMPL order
        remapped_joints = remap_joints(joint_positions)
        # Debug after remapping
        visualize_debug_joints(remapped_joints, "After Remapping", "static/debug_2_remapped.png")
        logger.info(f"Remapped joints stats - min: {remapped_joints.min()}, max: {remapped_joints.max()}")
        
        # Transform joints to match SMPL coordinate system
        transformed_joints = transform_input_joints(remapped_joints)
        # Debug after transformation
        visualize_debug_joints(transformed_joints, "After Transform", "static/debug_3_transformed.png")
        logger.info(f"Transformed joints stats - min: {transformed_joints.min()}, max: {transformed_joints.max()}")
        
        # Prepare input for pose network
        joints_tensor = torch.tensor(transformed_joints, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        joints_tensor = joints_tensor.to(device)
        
        # Get pose parameters from network
        with torch.no_grad():
            pose_params = pose_network(joints_tensor)
            pose_params = pose_params.cpu().numpy().squeeze()  # Remove batch dim

        # Get predicted joints from the pose parameters
        predicted_joints = pose_estimator.forward_kinematics(pose_params)
        # Remap predicted joints to frontend order
        predicted_joints = pose_estimator.remap_joints_to_frontend(predicted_joints)
        
        # Debug after prediction
        visualize_debug_joints(predicted_joints, "Predicted Joints", "static/debug_4_predicted.png")
        logger.info(f"Predicted joints stats - min: {predicted_joints.min()}, max: {predicted_joints.max()}")
        
        # Visualize both input and predicted joints in the same plot
        comparison_viz_path = visualize_joint_comparison(
            input_joints=joint_positions,
            predicted_joints=predicted_joints,
            selected_joint=selected_joint,
            title="static/joint_comparison.png"
        )
        
        frontend_pose_params = remap_pose_params_back(pose_params)
        glb_path = pose_estimator.export_pose_glb(pose_params, "static/optimized_pose_nets.glb")
        output_viz_path = pose_estimator.visualize_pose(pose_params=pose_params, title="static/optimized_pose_nets.png", selected_joint=selected_joint)

        result = {
            'pose_params': frontend_pose_params.tolist(),
            'predicted_joints': predicted_joints.tolist(),
            'comparison_viz': '/static/joint_comparison.png',
            'debug_viz': {
                'original': '/static/debug_1_original.png',
                'remapped': '/static/debug_2_remapped.png',
                'transformed': '/static/debug_3_transformed.png',
                'predicted': '/static/debug_4_predicted.png'
            },
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during pose estimation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/generate_from_keyframes', methods=['POST'])
def handle_generate_from_keyframes():
    try:
        data = request.get_json()
        logger.info("Received keyframe generation request")
        keyframes = data["keyframes"]
        number_diffusion_steps: int = int(data["numDiffusionSteps"])

        original_keyframes = data.get("originalKeyframes", [])
        motion_editing = True if original_keyframes else False
        # Extract frame indices
        keyframe_indices = [kf["frame"] for kf in keyframes]
        first_keyframe_index = keyframe_indices[0] if keyframe_indices else None
        
        # Determine the range to fill with original keyframes
        start_fill = max(0, first_keyframe_index - 20)
        end_fill = first_keyframe_index + 21  # 20 frames on either side + 1 for the keyframe itself
        
        # Initialize combined data
        max_frame =  196 if original_keyframes else max(kf['frame'] for kf in keyframes) + 1
        combined_data = np.zeros((max_frame, 263))  # Initialize with zeros
        
        # Process each keyframe
        for keyframe in keyframes:
            frame_idx = keyframe['frame']
            motion_data = keyframe['motionData']
            
            # Store global information
            combined_data[frame_idx, 0] = motion_data['global']['rootRotationDelta']  # 1 value
            combined_data[frame_idx, 1:3] = motion_data['global']['rootPositionDelta']  # 2 values
            combined_data[frame_idx, 3] = motion_data['global']['rootHeight']  # 1 value
            
            # Create remapped arrays with correct sizes
            positions = np.zeros((21, 3))  # 21 joints x 3 coordinates
            rotations = np.zeros((21, 6))  # 21 joints x 6 rotation values
            velocities = np.zeros((22, 3))  # 22 joints (including root) x 3 coordinates
            
            # Get original data
            orig_positions = np.array(motion_data['local']['jointPositions']).reshape(-1, 3)
            orig_rotations = np.array(motion_data['local']['jointRotations']).reshape(-1, 6)
            orig_velocities = np.array(motion_data['local']['jointVelocities']).reshape(-1, 3)
            
            # Remap positions and rotations using joint_mapping_no_hands_no_pelvis (21 joints)
            for frontend_idx, smpl_idx in joint_mapping_no_hands_no_pelvis.items():
                positions[smpl_idx] = orig_positions[frontend_idx] 
                rotations[smpl_idx] = orig_rotations[frontend_idx]  
            
            # Remap velocities using joint_mapping_no_hands (22 joints including root)
            for frontend_idx, smpl_idx in joint_mapping_no_hands.items():
                velocities[smpl_idx] = orig_velocities[frontend_idx]
            
            # Store remapped data
            combined_data[frame_idx, 4:67] = positions.reshape(-1)  # joint positions: 21*3 = 63 values
            combined_data[frame_idx, 67:193] = rotations.reshape(-1)  # joint rotations: 21*6 = 126 values
            combined_data[frame_idx, 193:259] = velocities.reshape(-1)  # joint velocities, including pelvis: 22*3 = 66 values
            
            foot_contact = motion_data['local']['footContact']  # 4 values
            combined_data[frame_idx, 259:263] = foot_contact
        
        # Fill the rest with original keyframes
        for original_keyframe in original_keyframes:
            frame_idx = original_keyframe['frame']
            if frame_idx > 195:
                break
            if start_fill <= frame_idx < end_fill:
                continue  # Skip the range around the first keyframe
            
            motion_data = original_keyframe['motionData']
            # Store global information
            combined_data[frame_idx, 0] = motion_data['global']['rootRotationDelta']  # 1 value
            combined_data[frame_idx, 1:3] = motion_data['global']['rootPositionDelta']  # 2 values
            combined_data[frame_idx, 3] = motion_data['global']['rootHeight']  # 1 value
            
            # Create remapped arrays with correct sizes
            positions = np.zeros((21, 3))  # 21 joints x 3 coordinates
            rotations = np.zeros((21, 6))  # 21 joints x 6 rotation values
            velocities = np.zeros((22, 3))  # 22 joints (including root) x 3 coordinates
            
            # Get original data
            orig_positions = np.array(motion_data['local']['jointPositions']).reshape(-1, 3)
            orig_rotations = np.array(motion_data['local']['jointRotations']).reshape(-1, 6)
            orig_velocities = np.array(motion_data['local']['jointVelocities']).reshape(-1, 3)
            
            # Remap positions and rotations using joint_mapping_no_hands_no_pelvis (21 joints)
            for frontend_idx, smpl_idx in joint_mapping_no_hands_no_pelvis.items():
                positions[smpl_idx] = orig_positions[frontend_idx] 
                rotations[smpl_idx] = orig_rotations[frontend_idx]  
            
            # Remap velocities using joint_mapping_no_hands (22 joints including root)
            for frontend_idx, smpl_idx in joint_mapping_no_hands.items():
                velocities[smpl_idx] = orig_velocities[frontend_idx]
            
            # Store remapped data
            combined_data[frame_idx, 4:67] = positions.reshape(-1)  # joint positions: 21*3 = 63 values
            combined_data[frame_idx, 67:193] = rotations.reshape(-1)  # joint rotations: 21*6 = 126 values
            combined_data[frame_idx, 193:259] = velocities.reshape(-1)  # joint velocities, including pelvis: 22*3 = 66 values
            
            foot_contact = motion_data['local']['footContact']  # 4 values
            combined_data[frame_idx, 259:263] = foot_contact

        # Save to .npy file
        output_path = 'static/motion_data.npy'
        np.save(output_path, combined_data)

        
        generated_motion = generate_inbetween_motion(combined_data, number_diffusion_steps, keyframe_indices,first_keyframe_index, motion_editing)[0][0]
        return jsonify({
            'status': 'success',
            'generated_motion': generated_motion.tolist() if generated_motion is not None else None
        })

    except Exception as e:
        logger.error(f"Error during generation from keyframes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/estimate_sequence', methods = ['POST'])
def handle_sequence_estimation():
    try:
        # We log that this process is starting
        logger.info("Received sequence estimation request")

        # let's take the joints from the animation into a ndarray
        sequence_joints = np.array(request.get_json()["jointPositions"]["allPositions"])[:,1:]  # (sequence_length, 24, 3)

        # since mdm uses a different input, we have to convert the joints into their format
        to_human3dml_tensor = prepare_smpl_for_priorMDM(sequence_joints)

        # We impaint the sequence again using priormdm
        sequence_predicted_joints = mini_prior_mdm(to_human3dml_tensor)

        return jsonify({
            'status': 'success',
            'generated_motion': sequence_predicted_joints.tolist() if sequence_predicted_joints is not None else None
        })

    except Exception as e:
        logger.error(f"Error during sequence estimation: {str(e)}", exc_info = True)
        return jsonify({'error': str(e)}), 500

def visualize_joint_comparison(input_joints, predicted_joints, selected_joint=None, title="joint_comparison.png"):
    """Visualize input and predicted joint positions in the same plot"""
    fig = plt.figure(figsize=(15, 5))

    # Create color arrays
    input_colors = ['blue'] * len(input_joints)
    pred_colors = ['red'] * len(predicted_joints)
    if selected_joint is not None:
        input_colors[selected_joint] = 'lime'
        pred_colors[selected_joint] = 'yellow'
    
    # Front view 
    ax1 = fig.add_subplot(131, projection='3d')
    for i, (joint, color) in enumerate(zip(input_joints, input_colors)):
        ax1.scatter(joint[0], joint[1], joint[2], c=color, marker='o', label='Input' if i == 0 else "")
    for i, (joint, color) in enumerate(zip(predicted_joints, pred_colors)):
        ax1.scatter(joint[0], joint[1], joint[2], c=color, marker='^', label='Predicted' if i == 0 else "")
    ax1.view_init(elev=90, azim=-90)
    ax1.set_title('Front View')
    
    # Side view
    ax2 = fig.add_subplot(132, projection='3d')
    for i, (joint, color) in enumerate(zip(input_joints, input_colors)):
        ax2.scatter(joint[0], joint[2], joint[1], c=color, marker='o')
    for i, (joint, color) in enumerate(zip(predicted_joints, pred_colors)):
        ax2.scatter(joint[0], joint[2], joint[1], c=color, marker='^')
    ax2.view_init(elev=0, azim=0)
    ax2.set_title('Side View')
    
    # Top view
    ax3 = fig.add_subplot(133, projection='3d')
    for i, (joint, color) in enumerate(zip(input_joints, input_colors)):
        ax3.scatter(joint[0], joint[1], joint[2], c=color, marker='o')
    for i, (joint, color) in enumerate(zip(predicted_joints, pred_colors)):
        ax3.scatter(joint[0], joint[1], joint[2], c=color, marker='^')
    ax3.view_init(elev=0, azim=-90)
    ax3.set_title('Top View')
    
    # Set consistent axes limits and labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_box_aspect([1,1,1])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(title)
    plt.close()
    
    return title

def remap_joints(input_joints):
        """
        Remap joints from frontend ordering to SMPL ordering
        
        Args:
            input_joints: numpy array of shape (24, 3) in frontend order
        
        Returns:
            remapped_joints: numpy array of shape (24, 3) in SMPL order
        """
        if input_joints.shape != (24, 3):
            raise ValueError(f"Expected input_joints shape (24, 3), got {input_joints.shape}")

        remapped_joints = np.zeros((24, 3))
        for frontend_idx, smpl_idx in joint_mapping.items():
            remapped_joints[smpl_idx] = input_joints[frontend_idx]
        return remapped_joints

def remap_pose_params_back(smpl_pose_params):
    """
    Remap pose parameters from SMPL ordering back to frontend ordering
    
    Args:
        smpl_pose_params: numpy array of shape (72,) in SMPL order
        (first 3 values are global orientation, then 23 joints * 3 rotation params)
    
    Returns:
        frontend_pose_params: numpy array of shape (72,) in frontend order
    """
    if smpl_pose_params.shape != (72,):
        raise ValueError(f"Expected smpl_pose_params shape (72,), got {smpl_pose_params.shape}")

    # Create output array
    frontend_pose_params = np.zeros(72)
    
    for frontend_idx, smpl_idx in joint_mapping.items():
        src_idx = smpl_idx * 3 
        dst_idx = frontend_idx * 3 
        frontend_pose_params[dst_idx:dst_idx+3] = smpl_pose_params[src_idx:src_idx+3]
    
    return frontend_pose_params

def transform_input_joints(joints):
    transformed_joints = joints.copy()
    transformed_joints[:, 1], transformed_joints[:, 2] = - joints[:, 2], joints[:, 1]
    return transformed_joints

def visualize_debug_joints(joints, title, save_path):
    """Debug visualization function for joint positions"""
    fig = plt.figure(figsize=(15, 5))
    
    # Front view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='b', marker='o')
    ax1.view_init(elev=90, azim=-90)
    ax1.set_title(f'Front View - {title}')
    
    # Side view
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='b', marker='o')
    ax2.view_init(elev=0, azim=0)
    ax2.set_title(f'Side View - {title}')
    
    # Top view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='b', marker='o')
    ax3.view_init(elev=0, azim=-90)
    ax3.set_title(f'Top View - {title}')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_box_aspect([1,1,1])
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 
