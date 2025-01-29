import React, { useRef, useState, useEffect } from 'react';
import { TransformControls } from '@react-three/drei';
import * as THREE from 'three';
import { useGLTF } from '@react-three/drei';
import { SkeletonHelper } from 'three';

const SMPLStatic = React.forwardRef(({ modelUrl, onJointSelect, selectedJoint, onJointPositionsUpdate }, ref) => {
    const groupRef = useRef();
    const [joints, setJoints] = useState([]);
    const [jointWorldPositions, setJointWorldPositions] = useState([]);
    const [draggedJointPositions, setDraggedJointPositions] = useState([]);
    const [initialTransforms, setInitialTransforms] = useState([]);
    const { scene } = useGLTF(modelUrl, true);
    const skeletonHelperRef = useRef();
    useEffect(() => {
        if (!scene || !groupRef.current) return;

        // Scale the model to a consistent size (2 units)
        const box = new THREE.Box3().setFromObject(scene);
        if (!box.isEmpty()) {
            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const desiredSize = 2.0;
            const scaleFactor = desiredSize / maxDim;
            scene.scale.set(scaleFactor, scaleFactor, scaleFactor);
            scene.updateMatrixWorld(true);
        }

        // Make the body semi-transparent
        scene.traverse((obj) => {
            if (obj.isMesh) {
                obj.material.transparent = true;
                obj.material.opacity = 0.5;
            }
        });

        groupRef.current.add(scene);

        // Find and setup skeleton helper
        let skinnedMeshFound = null;
        scene.traverse((obj) => {
            if (obj.isSkinnedMesh && !skinnedMeshFound) {
                skinnedMeshFound = obj;
            }
        });
        if (skinnedMeshFound) {
            skeletonHelperRef.current = new SkeletonHelper(skinnedMeshFound.skeleton.bones[0]);
            groupRef.current.add(skeletonHelperRef.current);
        }

        // Setup joints
        const foundJoints = [];
        const initialTransformsData = [];
        scene.traverse((obj) => {
            if (obj.isBone) {
                foundJoints.push({
                    name: obj.name,
                    bone: obj
                });
                // Store initial position and rotation
                initialTransformsData.push({
                    position: obj.position.clone(),
                    rotation: obj.rotation.clone(),
                    quaternion: obj.quaternion.clone()
                });
            }
        });
        setJoints(foundJoints);
        setInitialTransforms(initialTransformsData);
        console.log('SMPL Joint Order:', foundJoints.map((joint, index) => `${index}: ${joint.name}`));
        console.log('Loaded SMPL scene. Found joints:', foundJoints.length);

        // Expose scene and joints through ref
        if (ref) {
            ref.current = {
                scene,
                joints: foundJoints,
                resetJoints: () => {
                    // Restore each bone's initial position and rotation
                    foundJoints.forEach((joint, index) => {
                        joint.bone.position.copy(initialTransformsData[index].position);
                        joint.bone.quaternion.copy(initialTransformsData[index].quaternion);
                        joint.bone.updateMatrix();
                    });

                    // Update the scene and skeleton
                    scene.updateMatrixWorld(true);

                    // Force an update of joint positions after reset
                    const newPositions = foundJoints.map(joint =>
                        joint.bone.getWorldPosition(new THREE.Vector3())
                    );
                    setJointWorldPositions(newPositions);
                    setDraggedJointPositions([]);

                    // Notify parent about position update
                    onJointPositionsUpdate({
                        positions: foundJoints.map(joint =>
                            joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
                        ),
                        rotations: foundJoints.map(joint =>
                            joint.bone.quaternion.clone()
                        )
                    });
                }
            };
        }
    }, [scene, ref]);

    useEffect(() => {
        if (joints.length > 0) {
            const updatePositions = () => {
                // Force update the entire scene matrix
                scene.updateMatrixWorld(true);

                // SkeletonHelper updates automatically with the bones
                // No need to call update() on it

                const newPositions = joints.map(joint => {
                    const worldPos = new THREE.Vector3();
                    joint.bone.getWorldPosition(worldPos);
                    return worldPos;
                });
                setJointWorldPositions(newPositions);

                // Notify parent about position update
                onJointPositionsUpdate({
                    positions: joints.map(joint =>
                        joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
                    ),
                    rotations: joints.map(joint =>
                        joint.bone.quaternion.clone()
                    )
                });
            };

            // Set up animation frame for continuous updates
            let frameId;
            const animate = () => {
                updatePositions();
                frameId = requestAnimationFrame(animate);
            };
            frameId = requestAnimationFrame(animate);

            return () => {
                cancelAnimationFrame(frameId);
            };
        }
    }, [joints, scene]);

    const handleJointDrag = (index, newPosition) => {
        const updatedPositions = [...draggedJointPositions];
        updatedPositions[index] = newPosition;
        setDraggedJointPositions(updatedPositions);

        // Update joint world positions when dragging
        const newWorldPositions = joints.map(joint =>
            joint.bone.getWorldPosition(new THREE.Vector3())
        );
        setJointWorldPositions(newWorldPositions);

        const allJointPositions = joints.map(joint =>
            joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
        );
        onJointPositionsUpdate({
            positions: allJointPositions,
            rotations: joints.map(joint =>
                joint.bone.quaternion.clone()
            )
        });
    };

    // Add cleanup effect
    useEffect(() => {
        return () => {
            if (groupRef.current) {
                // Dispose of geometries and materials
                groupRef.current.traverse((object) => {
                    if (object.geometry) {
                        object.geometry.dispose();
                    }
                    if (object.material) {
                        if (Array.isArray(object.material)) {
                            object.material.forEach(material => material.dispose());
                        } else {
                            object.material.dispose();
                        }
                    }
                });

                // Remove everything from the group
                while (groupRef.current.children.length > 0) {
                    groupRef.current.remove(groupRef.current.children[0]);
                }
            }

            // Release GLTF resources
            if (scene) {
                useGLTF.clear(modelUrl);
            }
        };
    }, [modelUrl, scene]);

    return (
        <group ref={groupRef}>
            {joints.map((joint, index) => {
                const { bone } = joint;
                const isSelected = selectedJoint === index;
                const worldPos = jointWorldPositions[index] || new THREE.Vector3();

                const JointMesh = (
                    <mesh
                        onClick={(e) => {
                            e.stopPropagation();
                            onJointSelect(index);
                        }}
                    >
                        <sphereGeometry args={[0.02, 16, 16]} />
                        <meshStandardMaterial
                            color={isSelected ? '#ff0000' : '#ffffff'}
                        />
                    </mesh>
                );

                if (isSelected) {
                    return (
                        <group key={`joint-${index}`}>
                            <TransformControls
                                mode="translate"
                                object={bone}
                                onObjectChange={(e) => {
                                    handleJointDrag(index, e.target.object.getWorldPosition(new THREE.Vector3()));
                                }}
                            />
                            <group position={worldPos}>
                                {JointMesh}
                            </group>
                        </group>
                    );
                } else {
                    return (
                        <group
                            key={`joint-${index}`}
                            position={worldPos}
                        >
                            {JointMesh}
                        </group>
                    );
                }
            })}
        </group>
    );
});

export default SMPLStatic; 