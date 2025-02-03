import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import SMPLStatic from './SMPLStatic';
import BackButton from './BackButton';
import * as THREE from 'three';

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
    console.log(jointIndex);

    if (jointIndex < 0 || jointIndex >= Object.keys(SMPL_KINEMATIC_TREE).length) {
        return [];
    }
    return [jointIndex, ...SMPL_KINEMATIC_TREE[jointIndex]];
};

function SynthesisViewer({ onBack, onUpdatePoseRef, setAnimationInfo, setCurrentJointPositionCallback, sequencePositions, currentFrame }) {
    const MAX_KEYFRAME_LENGTH = 196;
    const [selectedJoint, setSelectedJoint] = useState(null);
    const [draggedJointIndices, setDraggedJointIndices] = useState([]);
    const [currentJointPositions, setCurrentJointPositions] = useState(null);
    const [modelUrl, setModelUrl] = useState(() => {
        // Check if we're running on GitHub Pages
        const isGitHubPages = window.location.hostname === 'ernaz100.github.io';
        if (isGitHubPages) {
            return 'https://ernaz100.github.io/dragbased_motion_editing/human.glb';
        }
        // Local development
        return '/human.glb';
    });
    const smplRef = useRef(null);
    const [pelvisOffset, setPelvisOffset] = useState(null);
    const handleJointSelect = (jointIndex) => {
        console.log('Selected joint:', jointIndex);
        setSelectedJoint(jointIndex);
    };
    useEffect(() => {
        setAnimationInfo(prevInfo => ({ ...prevInfo, numKeyframes: MAX_KEYFRAME_LENGTH }))
    }, [])

    const handleJointPositionsUpdate = ({ positions, rotations }) => {
        setCurrentJointPositionCallback({ positions, rotations }); //absolute positions
        setDraggedJointIndices(prev => {
            if (selectedJoint !== null && !prev.includes(selectedJoint)) {
                return [...prev, selectedJoint];
            }
            return prev;
        });
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
                console.log('Resetting specific joints...', draggedJointIndices);
                const jointIndicesToReset = draggedJointIndices
                    .flatMap(joint => [...getKinematicChain(joint), joint])
                    .filter(index => index >= 0 && index < smplRef.current.joints.length && ![0, 1].includes(index));
                console.log("toReset", jointIndicesToReset);

                smplRef.current.resetJoints(jointIndicesToReset);

                console.log('Applying new rotations...');
                // Then apply the new rotations
                smplRef.current.joints.forEach((joint, index) => {
                    if (!jointIndicesToReset.includes(index) || [0, 1].includes(index)) return; // Skip root and joints not in kinematic chain
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
                setDraggedJointIndices([])
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
                />
            </Canvas>
        </div>
    );
}

export default SynthesisViewer; 