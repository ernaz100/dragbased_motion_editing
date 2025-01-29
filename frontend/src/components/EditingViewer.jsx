import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import SMPLAnimated from './SMPLAnimated';
import BackButton from './BackButton';

export const BACKEND_URL = 'http://localhost:5001';

function EditingViewer({ currentTime, onAnimationLoaded, onUpdatePoseRef, setCurrentJointPositionCallback, setAllPositions, sequencePositions, isPlaying, onBack, currentFrame }) {
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

    const handleJointPositionsUpdate = ({ positions, rotations }) => {
        setCurrentJointPositionCallback({ positions, rotations }); //absolute positions
        const pelvisPos = positions[1];  // Get pelvis position
        setPelvisOffset(pelvisPos); // Store the pelvis offset for later use
        const relativePositions = positions.slice(1).map(pos => [
            pos[0] - pelvisPos[0],
            pos[1] - pelvisPos[1],
            pos[2] - pelvisPos[2]
        ]);
        setCurrentJointPositions(relativePositions);
    };

    const handleUpdatePose = async () => {
        if (!currentJointPositions || !pelvisOffset) {
            console.log('EditingViewer: No joint positions or pelvis offset available');
            return;
        }
        try {
            console.log('EditingViewer: Sending positions:', currentJointPositions);
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
                const jointIndicesToReset = [selectedJoint - 2, selectedJoint - 1, selectedJoint, selectedJoint + 1, selectedJoint + 2]
                    .filter(index => index >= 0 && index < smplRef.current.joints.length && ![0, 1].includes(index));

                smplRef.current.resetJoints(jointIndicesToReset);

                console.log('Applying new rotations...');
                // Then apply the new rotations
                smplRef.current.joints.forEach((joint, index) => {
                    if (![selectedJoint - 2, selectedJoint - 1, selectedJoint, selectedJoint + 1, selectedJoint + 2].includes(index) || [0, 1].includes(index)) return
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

            // After successful pose update and skeleton update
            if (smplRef.current && smplRef.current.scene) {
                // Get the updated joint positions after pose change
                const updatedPositions = smplRef.current.joints.map(joint => {
                    const worldPosition = new THREE.Vector3();
                    joint.bone.getWorldPosition(worldPosition);
                    return [worldPosition.x, worldPosition.y, worldPosition.z];
                });

                // Get the updated rotations
                const updatedRotations = smplRef.current.joints.map(joint => {
                    const quaternion = joint.bone.quaternion.clone();
                    return [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
                });

                // Update the current positions
                setCurrentJointPositionCallback({
                    positions: updatedPositions,
                    rotations: updatedRotations
                });
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
                        mode="editing"
                        setAllPositions={setAllPositions}
                    />
                )}
                < OrbitControls
                    makeDefault
                    enabled={selectedJoint === null}
                />
            </Canvas>
        </div>
    );
}

export default EditingViewer; 