import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import SMPLAnimated from './SMPLAnimated'; // Import the SMPLModel component
import poseParams from './pose_params.json';





export const BACKEND_URL = 'http://localhost:5001';
function ModelViewer({ currentTime, onAnimationLoaded, onUpdatePoseRef, onUpdateAnimation, sequencePositions, isPlaying }) {
    const [selectedJoint, setSelectedJoint] = useState(null);
    const [currentJointPositions, setCurrentJointPositions] = useState(null);
    const [modelUrl, setModelUrl] = useState('/human.glb');
    const smplRef = useRef(null);
    const [pelvisOffset, setPelvisOffset] = useState(null);

    const handleJointSelect = (jointIndex) => {
        console.log('Selected joint:', jointIndex);
        setSelectedJoint(jointIndex);
    };
    const handleUpdateAnimation = (positions) => {
        onUpdateAnimation(positions);
    }
    const handleJointPositionsUpdate = (positions) => {
        const pelvisPos = positions[1];  // Get pelvis position
        setPelvisOffset(pelvisPos); // Store the pelvis offset for later use
        const relativePositions = positions.slice(1).map(pos => [
            pos[0] - pelvisPos[0],
            pos[1] - pelvisPos[1],
            pos[2] - pelvisPos[2]
        ]);
        setCurrentJointPositions(relativePositions);
    };
    useEffect(() => {
        handleUpdateSequence(sequencePositions)
    }, [sequencePositions])

    const handleUpdateSequence = (sequencePositions) => {
        console.log('Handling sequence positions:', sequencePositions);
        if (smplRef.current && smplRef.current.updateSequence) {
            smplRef.current.updateSequence(sequencePositions);
        }
    };

    const handleUpdatePose = async () => {
        if (!currentJointPositions || !pelvisOffset) {
            console.log('ModelViewer: No joint positions or pelvis offset available');
            return;
        }
        try {
            console.log('ModelViewer: Sending positions:', currentJointPositions);
            const response = await fetch(`${BACKEND_URL}/estimate_pose`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    joint_positions: currentJointPositions,
                    selected_joint: selectedJoint - 1,
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received response:', data);

            const { pose_params } = data;

            // Update the joint positions in the scene
            if (smplRef.current && smplRef.current.joints && smplRef.current.resetJoints) {
                smplRef.current.resetJoints();
                smplRef.current.joints.forEach((joint, index) => {
                    if (index === 0) return; // Skip root joint
                    if (index < 25) { // SMPL has 24 joints (excluding root)
                        const baseIdx = (index - 1) * 3;
                        const axisAngleRotation = new THREE.Vector3(
                            pose_params[baseIdx],
                            pose_params[baseIdx + 1],
                            pose_params[baseIdx + 2]
                        );
                        // Convert axis-angle to quaternion
                        const angle = axisAngleRotation.length();
                        const quaternion = new THREE.Quaternion();
                        if (angle > 0.0001) { // Avoid division by zero
                            axisAngleRotation.normalize();
                            quaternion.setFromAxisAngle(axisAngleRotation, angle);
                        }
                        // Apply the rotation
                        joint.bone.quaternion.copy(quaternion);
                    }
                });

                // Update the skeleton
                smplRef.current.scene.updateMatrixWorld(true);
            }
        } catch (error) {
            console.error('Error updating pose:', error);
        }
    };

    useEffect(() => {
        if (onUpdatePoseRef) {
            onUpdatePoseRef(handleUpdatePose);
        }
    }, [onUpdatePoseRef, currentJointPositions]);

    return (
        <Canvas
            camera={{ position: [0, 1, 3], fov: 60 }}
            style={{ background: '#1a1a1a' }}
        >
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <SMPLAnimated
                ref={smplRef}
                modelUrl={modelUrl}
                onJointSelect={handleJointSelect}
                animationTime={currentTime}
                selectedJoint={selectedJoint}
                onAnimationLoaded={onAnimationLoaded}
                onJointPositionsUpdate={handleJointPositionsUpdate}
                onUpdateAnimation={handleUpdateAnimation}
                isPlaying={isPlaying}
            />
            <OrbitControls
                makeDefault
                enabled={selectedJoint === null}
            />
        </Canvas>
    );
}

export default ModelViewer; 