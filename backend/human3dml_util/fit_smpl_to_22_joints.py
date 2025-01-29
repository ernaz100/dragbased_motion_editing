import numpy as np
import torch
from smplx import SMPL

# Define the function to fit SMPL models to 22-joint sequences
def fit_smpl_to_22_joints(joint_sequences, device):
    """
    Fits SMPL model to a sequence of 22-joint positions.

    Args:
        joint_sequences (numpy array): Sequence of 22 joint positions with shape (N, 22, 3),
                                       where N is the number of frames.
        smpl_model_path (str): Path to the SMPL model folder.

    Returns:
        list: A list of SMPL parameters (pose, shape, translation) for each frame.
    """
    # Load SMPL model
    smpl = SMPL(model_path="../models/smpl/SMPL_NEUTRAL.pkl", gender='neutral', batch_size=1).to(device)

    # Prepare outputs
    smpl_params_sequence = []

    # Optimization parameters
    num_joints = 22
    joint_weights = torch.ones(num_joints, device=device)  # Equal weights for all joints

    for joints in joint_sequences.unbind(0):
        # Convert joints to torch tensor
        joints = joints.unsqueeze(0)  # Shape (1, 22, 3)

        # Initialize SMPL parameters
        pose = torch.zeros(1, 72, requires_grad=True, device=device)  # SMPL pose (24 joints * 3 angles)
        shape = torch.zeros(1, 10, requires_grad=True, device=device)  # SMPL shape (10 coefficients)
        translation = torch.zeros(1, 3, requires_grad=True, device=device)  # Global translation

        optimizer = torch.optim.Adam([pose, shape, translation], lr=0.01)

        # Optimization loop
        for i in range(100):  # 100 iterations
            optimizer.zero_grad()

            # Forward pass through SMPL model
            smpl_output = smpl.forward(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3],
                                       transl=translation)

            smpl_joints = smpl_output.joints[:, :num_joints]  # Extract the first 22 joints

            # Compute loss (mean squared error between input and SMPL joints)
            loss = torch.mean(joint_weights * (smpl_joints - joints) ** 2)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Save optimized parameters
        smpl_params_sequence.append({
            "pose": pose.detach().cpu().numpy(),
            "shape": shape.detach().cpu().numpy(),
            "translation": translation.detach().cpu().numpy()
        })

    return smpl_params_sequence

# Example usage
# joint_sequences = np.random.rand(10, 22, 3)  # Example input: 10 frames, 22 joints, 3D positions
# smpl_model_path = "path/to/smpl/model"  # Path to SMPL model files
# smpl_params = fit_smpl_to_22_joints(joint_sequences, smpl_model_path)