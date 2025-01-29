import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import SMPLStatic from './SMPLStatic';
import BackButton from './BackButton';
import * as THREE from 'three';

export const BACKEND_URL = 'http://localhost:5001';

function SynthesisViewer({ onBack, onUpdatePoseRef, setAnimationInfo, setCurrentJointPositionCallback, sequencePositions, currentFrame }) {
    const MAX_KEYFRAME_LENGTH = 196;
    const [selectedJoint, setSelectedJoint] = useState(null);
    const [currentJointPositions, setCurrentJointPositions] = useState(null);
    const [modelUrl, setModelUrl] = useState('/human.glb');
    const smplRef = useRef(null);
    const [pelvisOffset, setPelvisOffset] = useState(null);
    const keyframes = Array(MAX_KEYFRAME_LENGTH).fill(0);
    const handleJointSelect = (jointIndex) => {
        console.log('Selected joint:', jointIndex);
        setSelectedJoint(jointIndex);
    };
    useEffect(() => {
        setAnimationInfo(prevInfo => ({ ...prevInfo, numKeyframes: MAX_KEYFRAME_LENGTH }))
    }, [])

    const handleJointPositionsUpdate = ({ positions, rotations }) => {
        setCurrentJointPositionCallback({ positions, rotations }); //absolute positions
        const pelvisPos = positions[1];  // Get pelvis position
        setPelvisOffset(pelvisPos); // Store the pelvis offset for later use

        // Convert positions to array if it's not already
        const positionsArray = Array.from(positions);
        const relativePositions = positionsArray.slice(1).map(pos => [
            pos[0] - pelvisPos[0],
            pos[1] - pelvisPos[1],
            pos[2] - pelvisPos[2]
        ]);
        setCurrentJointPositions(relativePositions);
    };

    const handleUpdatePose = async () => {
        if (!currentJointPositions || !pelvisOffset) {
            console.log('SynthesisViewer: No joint positions or pelvis offset available');
            return;
        }
        try {
            console.log('SynthesisViewer: Sending positions:', currentJointPositions);
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
            console.log('Received pose_params:', data.pose_params);

            const { pose_params } = data;

            // First reset the joints to their initial pose
            if (smplRef.current && smplRef.current.resetJoints) {
                console.log('Resetting specific joints...');
                const jointIndicesToReset = [selectedJoint - 2, selectedJoint - 1, selectedJoint, selectedJoint + 1, selectedJoint + 2]
                    .filter(index => index >= 0 && index < smplRef.current.joints.length && ![0, 1].includes(index));

                smplRef.current.resetJoints(jointIndicesToReset);

                console.log('Applying new rotations...');
                // Then apply the new rotations
                smplRef.current.joints.forEach((joint, index) => {
                    if (![selectedJoint - 2, selectedJoint - 1, selectedJoint, selectedJoint + 1, selectedJoint + 2].includes(index) || [0, 1].includes(index)) return; // Skip root and first few joints
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

                            // For pelvis (index 1), add an additional 90-degree rotation around X
                            if (index === 1) {
                                const pelvisCorrection = new THREE.Quaternion().setFromEuler(
                                    new THREE.Euler(-Math.PI / 2, 0, 0)
                                );
                                quaternion.multiply(pelvisCorrection);
                            }

                            // Apply the rotation
                            joint.bone.quaternion.copy(quaternion);
                        }
                        joint.bone.updateMatrix();
                        joint.bone.updateMatrixWorld(true);
                    }
                });

                // Update the entire scene hierarchy
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

    // Add AnimatedJoints component
    function AnimatedJoints({ positions }) {
        return positions.map((pos, idx) => (
            <mesh key={idx} position={[pos[0], pos[1], pos[2]]}>
                <sphereGeometry args={[0.03]} />
                <meshStandardMaterial
                    color={selectedJoint === idx ? '#ff0000' : '#ffffff'}
                    emissive={selectedJoint === idx ? '#ff0000' : '#000000'}
                />
            </mesh>
        ));
    }

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <BackButton onClick={onBack} />
            <Canvas
                camera={{ position: [0, 1, 3], fov: 60 }}
                style={{ background: '#1a1a1a' }}
            >
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                {sequencePositions && sequencePositions.length > 0 ? (
                    <AnimatedJoints positions={sequencePositions[currentFrame] || sequencePositions[0]} />
                ) : (
                    <SMPLStatic
                        ref={smplRef}
                        modelUrl={modelUrl}
                        onJointSelect={handleJointSelect}
                        selectedJoint={selectedJoint}
                        onJointPositionsUpdate={handleJointPositionsUpdate}
                    />
                )}
                <OrbitControls
                    makeDefault
                    enabled={selectedJoint === null}
                />
            </Canvas>
        </div>
    );
}

export default SynthesisViewer; 