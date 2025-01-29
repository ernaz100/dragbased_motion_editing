import sys
sys.path.append('/Users/ericnazarenus/Desktop/dragbased')
import torch
from priormdm.model import PriorMDM
from priormdm.config import Config
from priormdm.data_loaders.humanml.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

class MotionEditor:
    def __init__(self, checkpoint_path):
        """
        Initialize the motion editor with PriorMDM
        
        Args:
            checkpoint_path: Path to PriorMDM checkpoint
        """
        # Load config and model
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = PriorMDM(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def prepare_motion_input(self, smpl_poses, fps=30):
        """
        Convert SMPL poses to PriorMDM input format
        
        Args:
            smpl_poses: [seq_len, 72] SMPL pose parameters
            fps: Frames per second of the motion
        """
        # Convert axis-angle to rotation matrices
        poses_rot_mat = axis_angle_to_matrix(smpl_poses.reshape(-1, 3)).reshape(len(smpl_poses), 24, 3, 3)
        
        # Prepare motion input in PriorMDM format
        motion_input = {
            'poses': poses_rot_mat,
            'fps': torch.tensor([fps]),
            'lengths': torch.tensor([len(smpl_poses)])
        }
        
        return motion_input

    def edit_motion(self, original_motion, constraint_frame, target_joints, num_samples=1):
        """
        Edit motion using PriorMDM
        
        Args:
            original_motion: [seq_len, 72] tensor of SMPL pose parameters
            constraint_frame: Frame index to modify
            target_joints: [24, 3] tensor of target joint positions
            num_samples: Number of motion samples to generate
            
        Returns:
            List of edited motions
        """
        # Prepare input motion
        motion_input = self.prepare_motion_input(original_motion)
        
        # Create editing mask (1 for frames to keep, 0 for frames to edit)
        mask = torch.ones(len(original_motion))
        edit_window = 10  # Number of frames around constraint to edit
        start_frame = max(0, constraint_frame - edit_window)
        end_frame = min(len(original_motion), constraint_frame + edit_window)
        mask[start_frame:end_frame] = 0
        
        # Prepare conditioning
        conditioning = {
            'motion': motion_input,
            'mask': mask,
            'target_joints': target_joints,
            'target_frame': torch.tensor([constraint_frame])
        }
        
        # Generate samples
        with torch.no_grad():
            edited_motions = []
            for _ in range(num_samples):
                # Generate edited motion
                output = self.model.generate(
                    conditioning,
                    sample_fn=self.model.euler_maruyama_sample
                )
                
                # Convert output rotations back to axis-angle
                edited_motion = matrix_to_axis_angle(output['poses'].reshape(-1, 3, 3)).reshape(len(original_motion), 72)
                edited_motions.append(edited_motion)
        
        return edited_motions

def visualize_edit(original_motion, edited_motion, constraint_frame, pose_estimator):
    """
    Visualize original and edited motions
    """
    # Convert to joint positions
    original_joints = poses_to_joints(original_motion, pose_estimator)
    edited_joints = poses_to_joints(edited_motion, pose_estimator)
    
    # Create visualization directory
    os.makedirs('motion_edits', exist_ok=True)
    
    # Visualize frames around the edit
    window = 30  # frames to visualize before and after edit
    start_frame = max(0, constraint_frame - window)
    end_frame = min(len(original_motion), constraint_frame + window)
    
    for t in range(start_frame, end_frame):
        visualize_sequence(
            original_joints[t:t+1],
            edited_joints[t:t+1],
            t,
            'motion_edits'
        )
    
    # Create gif
    create_gif('motion_edits', 'motion_edit.gif')

# Example usage
if __name__ == "__main__":
    # Initialize motion editor
    editor = MotionEditor('path_to_priormdm_checkpoint.pt')
    
    # Load your SMPL motion
    original_motion = torch.load('your_motion.pt')  # [seq_len, 72] SMPL poses
    
    # Define target joint positions for editing
    target_joints = torch.randn(24, 3)  # Replace with your target joints
    constraint_frame = 30  # Frame to edit
    
    # Generate edited motion
    edited_motions = editor.edit_motion(
        original_motion,
        constraint_frame,
        target_joints,
        num_samples=3
    )
    
    # Visualize the edit
    visualize_edit(
        original_motion,
        edited_motions[0],  # Use first sample
        constraint_frame,
        pose_estimator
    ) 