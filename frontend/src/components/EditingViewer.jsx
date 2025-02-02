import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import SMPLAnimated from './SMPLAnimated';
import BackButton from './BackButton';

export const BACKEND_URL = 'http://localhost:5001';

const SMPL_KINEMATIC_TREE = {
    0: [],           // global root
    1: [0],          // pelvis
    2: [1, 0],       // left hip
    3: [2, 1, 0],    // left knee
    4: [3, 2, 1, 0], // left ankle
    5: [4, 3, 2, 1, 0], // left foot
    6: [1, 0],       // right hip
    7: [6, 1, 0],    // right knee
    8: [7, 6, 1, 0], // right ankle
    9: [8, 7, 6, 1, 0], // right foot
    10: [1, 0],      // spine1
    11: [10, 1, 0],  // spine2
    12: [11, 10, 1, 0], // spine3
    13: [12, 11, 10, 1, 0], // neck
    14: [13, 12, 11, 10, 1, 0], // head
    15: [12, 11, 10, 1, 0],     // left collar
    16: [15, 12, 11, 10, 1, 0], // left shoulder
    17: [16, 15, 12, 11, 10, 1, 0], // left elbow
    18: [17, 16, 15, 12, 11, 10, 1, 0], // left wrist
    19: [18, 17, 16, 15, 12, 11, 10, 1, 0], // left hand
    20: [12, 11, 10, 1, 0],     // right collar
    21: [20, 12, 11, 10, 1, 0], // right shoulder
    22: [21, 20, 12, 11, 10, 1, 0], // right elbow
    23: [22, 21, 20, 12, 11, 10, 1, 0], // right wrist
    24: [23, 22, 21, 20, 12, 11, 10, 1, 0], // right hand
};

const getKinematicChain = (jointIndex) => {
    if (jointIndex < 0 || jointIndex >= Object.keys(SMPL_KINEMATIC_TREE).length) {
        return [];
    }
    return [jointIndex, ...SMPL_KINEMATIC_TREE[jointIndex]];
};

function EditingViewer({ currentTime, onAnimationLoaded, onUpdatePoseRef, setCurrentJointPositionCallback, setAllPositions, sequencePositions, isPlaying, onBack, currentFrame }) {
    const [selectedJoint, setSelectedJoint] = useState(null);
    const [draggedJointIndices, setDraggedJointIndices] = useState([]);
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
        setDraggedJointIndices(prev => {
            if (selectedJoint !== null && !prev.includes(selectedJoint)) {
                return [...prev, selectedJoint];
            }
            return prev;
        });
        const pelvisPos = positions[1];
        setPelvisOffset(pelvisPos);
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

            if (smplRef.current && smplRef.current.resetJoints) {
                const jointIndicesToReset = draggedJointIndices
                    .flatMap(joint => [...getKinematicChain(joint), joint])
                    .filter(index => index >= 0 && index < smplRef.current.joints.length && ![0, 1].includes(index));

                smplRef.current.resetJoints(jointIndicesToReset);

                console.log('Applying new rotations...');
                smplRef.current.joints.forEach((joint, index) => {
                    if (!jointIndicesToReset.includes(index) || [0, 1].includes(index)) return;
                    if (index < 25) {
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

                setDraggedJointIndices([]); // Reset dragged joints after applying changes
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