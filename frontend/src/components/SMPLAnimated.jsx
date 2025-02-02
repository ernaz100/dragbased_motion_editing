import React, { useRef, useState, useEffect } from 'react';
import { TransformControls, useGLTF, useAnimations } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SkeletonHelper } from 'three';

const SMPLAnimated = React.forwardRef(({ modelUrl, onJointSelect, animationTime, selectedJoint, onAnimationLoaded, onJointPositionsUpdate, setAllPositions }, ref) => {
    const groupRef = useRef();
    const skeletonHelperRef = useRef();
    const [joints, setJoints] = useState([]);
    const [draggedJointPositions, setDraggedJointPositions] = useState([]);
    const [jointWorldPositions, setJointWorldPositions] = useState([]);
    const { scene, animations } = useGLTF(modelUrl, true);
    const { actions, names, mixer } = useAnimations(animations, groupRef);
    const [currentAction, setCurrentAction] = useState(null);
    const animationTimeRef = useRef(0);
    // Update the ref whenever animationTime changes, without causing re-renders
    useEffect(() => {
        animationTimeRef.current = animationTime;
    }, [animationTime]);

    useEffect(() => {
        if (!scene || !groupRef.current) return;

        // Add check for animations array and handle static models
        if (!animations || animations.length === 0) {
            console.log('No animations found in the model - loading as static mesh');
            onAnimationLoaded({
                duration: 0,
                numKeyframes: 0,
                framesPerSecond: 0
            });
        } else {
            const animation = animations[0];
            const quaternionTrack = animation.tracks.find(track => track.name.endsWith('.quaternion') && !track.name.includes('root'));
            const numKeyframes = 196 //quaternionTrack?.times.length || 0;
            const duration = animation.duration;
            console.log(animation.tracks);
            onAnimationLoaded({
                duration,
                numKeyframes,
                framesPerSecond: quaternionTrack?.times.length / duration
            });
        }
        // Automatically scale the model to a consistent size (2 units)
        // This ensures models display at a reasonable scale regardless of their original dimensions
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

        // Make the body semi-transparent or wireframe
        scene.traverse((obj) => {
            if (obj.isMesh) {
                obj.material.transparent = true;
                obj.material.opacity = 0.5;
                // Uncomment this for wireframe mode:
                // obj.material.wireframe = true;
            }
        });

        groupRef.current.add(scene);

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

        const foundJoints = [];
        scene.traverse((obj) => {
            if (obj.isBone) {
                foundJoints.push({
                    name: obj.name,
                    bone: obj
                });
            }
        });
        setJoints(foundJoints);
        console.log('SMPL Joint Order:', foundJoints.map((joint, index) => `${index}: ${joint.name}`));
        console.log('Loaded SMPL scene. Found joints:', foundJoints.length);
        // 0  : 0,0,0 ; 1 : 1.47,0.34,0.91
        // Only play animation if it exists
        if (names.length > 0) {
            const action = actions[names[0]];
            action.reset();
            action.clampWhenFinished = true;
            action.loop = THREE.LoopOnce;
            action.play();
        }

        // Expose scene, joints, and reset function through ref
        if (ref) {
            ref.current = {
                scene,
                joints: foundJoints,
                resetJoints: (jointIndicesToReset) => {
                    if (animations && animations.length > 0) {
                        const animation = animations[0];
                        foundJoints.forEach((joint, index) => {
                            // Skip if this joint is not in the reset list
                            if (!jointIndicesToReset.includes(index)) return;

                            // Find position track for this joint
                            const positionTrack = animation.tracks.find(
                                track => track.name === `${joint.bone.name}.position`
                            );

                            if (positionTrack) {
                                // Find the keyframes before and after current time
                                const times = positionTrack.times;
                                const values = positionTrack.values;

                                // Find the index of the keyframe just before current time
                                let index = times.findIndex(t => t > animationTime) - 1;
                                if (index < 0) index = 0;
                                if (index >= times.length - 1) index = times.length - 2;

                                // Get the two keyframe times and positions
                                const t1 = times[index];
                                const t2 = times[index + 1];
                                const p1 = new THREE.Vector3(
                                    values[index * 3],
                                    values[index * 3 + 1],
                                    values[index * 3 + 2]
                                );
                                const p2 = new THREE.Vector3(
                                    values[(index + 1) * 3],
                                    values[(index + 1) * 3 + 1],
                                    values[(index + 1) * 3 + 2]
                                );

                                // Linear interpolation
                                const alpha = (animationTime - t1) / (t2 - t1);
                                const position = p1.lerp(p2, alpha);

                                joint.bone.position.copy(position);
                            }

                            // Also reset quaternion if available
                            const quaternionTrack = animation.tracks.find(
                                track => track.name === `${joint.bone.name}.quaternion`
                            );

                            if (quaternionTrack) {
                                const times = quaternionTrack.times;
                                const values = quaternionTrack.values;

                                let index = times.findIndex(t => t > animationTime) - 1;
                                if (index < 0) index = 0;
                                if (index >= times.length - 1) index = times.length - 2;

                                const t1 = times[index];
                                const t2 = times[index + 1];
                                const q1 = new THREE.Quaternion(
                                    values[index * 4],
                                    values[index * 4 + 1],
                                    values[index * 4 + 2],
                                    values[index * 4 + 3]
                                );
                                const q2 = new THREE.Quaternion(
                                    values[(index + 1) * 4],
                                    values[(index + 1) * 4 + 1],
                                    values[(index + 1) * 4 + 2],
                                    values[(index + 1) * 4 + 3]
                                );

                                // Spherical linear interpolation for quaternions
                                const alpha = (animationTime - t1) / (t2 - t1);
                                const quaternion = q1.slerp(q2, alpha);

                                joint.bone.quaternion.copy(quaternion);
                            }
                        });
                        scene.updateMatrixWorld(true);
                    }
                },
                updateSequence: (sequencePositions) => {
                    const startTime = animationTimeRef.current;
                    // sequencePositions: array[30] of array[72] (24 joints × 3 axis angle representation)
                    const sequenceTracks = Array(24).fill().map((_, jointIndex) => {
                        // For each joint, collect its axis-angle coordinates across all frames
                        return sequencePositions.map(frame => {
                            const baseIdx = jointIndex * 3;
                            const axisAngle = new THREE.Vector3(
                                frame[baseIdx],     // x
                                frame[baseIdx + 1], // y
                                frame[baseIdx + 2]  // z
                            );
                            // Convert axis-angle to quaternion
                            const angle = axisAngle.length();
                            const quaternion = new THREE.Quaternion();
                            if (angle !== 0) {
                                const axis = axisAngle.normalize();
                                quaternion.setFromAxisAngle(axis, angle);
                            }
                            return {
                                quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
                            };
                        });
                    });

                    if (animations && animations[0]) {
                        const animation = animations[0];
                        const frameTime = 1 / 24; // 24 fps

                        // Update each joint's rotation and position tracks
                        for (let joint = 1; joint < 25; joint++) {
                            // Handle quaternion track
                            const quatTrackName = `${foundJoints[joint].bone.name}.quaternion`;
                            const quatTrack = animation.tracks.find(t => t.name === quatTrackName);
                            if (quatTrack) {
                                // Create new arrays for times and values
                                const newTimes = [...quatTrack.times];
                                const newQuatValues = [...quatTrack.values];

                                // Insert new keyframes at the correct position
                                sequenceTracks[joint - 1].forEach((frame, frameIndex) => {
                                    const insertTime = startTime + frameIndex * frameTime;
                                    const timeIndex = newTimes.findIndex(t => t >= insertTime);

                                    if (timeIndex === -1) {
                                        // Append to end if time is beyond existing timeline
                                        newTimes.push(insertTime);
                                        newQuatValues.push(...frame.quaternion);
                                    } else {
                                        // Shift all subsequent times forward by frameTime
                                        for (let i = timeIndex; i < newTimes.length; i++) {
                                            newTimes[i] += frameTime;
                                        }
                                        // Insert at appropriate time index
                                        newTimes.splice(timeIndex, 0, insertTime);
                                        newQuatValues.splice(timeIndex * 4, 0, ...frame.quaternion);
                                    }
                                });

                                quatTrack.times = new Float32Array(newTimes);
                                quatTrack.values = new Float32Array(newQuatValues);
                            }
                        }

                        // After updating other joints, get the new maximum duration
                        const quaternionTrack = animation.tracks.find(track =>
                            track.name.endsWith('.quaternion') && !track.name.includes('root')
                        );
                        const maxDuration = Math.max(...quaternionTrack.times);

                        // Update only the times for root joint tracks and position tracks
                        const rootTracks = animation.tracks.filter(track =>
                            track.name.startsWith(foundJoints[0].bone.name)
                        );

                        rootTracks.forEach(track => {
                            if (track.times.length > 0) {
                                track.times[track.times.length - 1] = maxDuration
                            }
                        });
                        const posTracks = animation.tracks.filter(track =>
                            track.name.endsWith('.position')
                        );
                        posTracks.forEach(track => {
                            if (track.times.length > 0) {
                                track.times[track.times.length - 1] = maxDuration
                            }
                        });

                        // Update animation duration and keyframe info
                        const numKeyframes = quaternionTrack?.times.length || 0;
                        const duration = Math.max(...quaternionTrack.times);
                        onAnimationLoaded({
                            duration,
                            numKeyframes,
                            framesPerSecond: numKeyframes / duration
                        });
                        // Stop and remove current animation before modifying tracks
                        if (currentAction) {
                            currentAction.stop();
                            mixer.uncacheAction(currentAction);
                        }
                        if (names.length > 0) {
                            actions[names[0]].stop();
                            mixer.uncacheAction(actions[names[0]]);
                        }
                        try {
                            // Validate and clean up tracks before creating new clip
                            const validTracks = animation.tracks.filter(track => {
                                // Check if track has valid times and values arrays
                                return track.times && track.times.length > 0 &&
                                    track.values && track.values.length > 0 &&
                                    // Ensure values array length matches expected size based on track type
                                    ((track.name.endsWith('.quaternion') && track.values.length === track.times.length * 4) ||
                                        (track.name.endsWith('.position') && track.values.length === track.times.length * 3) ||
                                        (track.name.endsWith('.scale') && track.values.length === track.times.length * 3));
                            });

                            // Create new clip with validated tracks
                            const newClip = new THREE.AnimationClip('modified_animation', duration, validTracks);

                            // Create and play new animation
                            const newAction = mixer.clipAction(newClip);
                            newAction.reset();
                            newAction.play();
                            setCurrentAction(newAction);
                        } catch (error) {
                            console.error('Animation creation failed:', error);
                            console.error('Error details:', {
                                trackCount: animation.tracks.length,
                                duration: duration,
                                firstTrack: animation.tracks[0]
                            });
                        }
                    }
                }
            };
        }

        // Add this new block to collect all positions
        if (animations && animations.length > 0) {
            const animation = animations[0];
            const duration = animation.duration;
            const quaternionTracks = animation.tracks.find(track => track.name.endsWith('.quaternion') && !track.name.includes('root'));
            const numKeyframes = quaternionTracks?.times.length || 0;

            const fps = numKeyframes / duration;
            const totalFrames = Math.floor(duration * fps);
            const allPositions = [];

            // Sample positions at each frame
            for (let frame = 0; frame <= totalFrames; frame++) {
                const time = (frame / totalFrames) * duration;

                // Update the animation to this time
                mixer.setTime(time);
                scene.updateMatrixWorld(true);

                // Collect positions for all joints at this frame
                const framePositions = foundJoints.map(joint =>
                    joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
                );
                allPositions.push(framePositions);
            }

            // Reset mixer time
            mixer.setTime(0);

            // Send all positions to parent
            if (setAllPositions) {
                setAllPositions(allPositions);
            }
        }
    }, [scene, animations, onAnimationLoaded, ref]);

    useFrame((state, delta) => {
        if (currentAction) {
            currentAction.paused = true;
            currentAction.time = animationTime;
        } else if (names.length > 0) {
            // Fallback to original animation if no custom animation exists
            const action = actions[names[0]];
            if (action) {
                action.paused = true;
                action.time = animationTime;
            }
        }
        // Always update joint positions
        if (joints.length > 0) {
            const newPositions = joints.map(joint =>
                joint.bone.getWorldPosition(new THREE.Vector3())
            );
            setJointWorldPositions(newPositions);
        }
    });

    const handleJointDrag = (index, newPosition) => {
        const updatedPositions = [...draggedJointPositions];
        updatedPositions[index] = newPosition;
        setDraggedJointPositions(updatedPositions);
        console.log(newPosition);
        // Collect all joint positions and send to parent
        const allJointPositions = joints.map(joint =>
            joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
        );
        onJointPositionsUpdate({
            positions: allJointPositions,
            rotations: joints.map(joint =>
                joint.bone.getWorldPosition(new THREE.Vector3()).toArray()
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

                // Clear animations and mixer
                if (mixer) {
                    mixer.stopAllAction();
                    mixer.uncacheRoot(scene);
                }

                // Remove everything from the group
                while (groupRef.current.children.length > 0) {
                    groupRef.current.remove(groupRef.current.children[0]);
                }
            }

            // Clear skeleton helper
            if (skeletonHelperRef.current) {
                skeletonHelperRef.current.dispose();
            }

            // Release GLTF resources
            if (scene) {
                useGLTF.clear(modelUrl);
            }
        };
    }, [modelUrl, scene, mixer]);

    return (
        <group ref={groupRef}>
            {joints.map((joint, index) => {
                const { bone } = joint; // The actual Bone object
                const isSelected = selectedJoint === index;
                const worldPos = jointWorldPositions[index] || new THREE.Vector3();

                // This mesh is how the user clicks the joint
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

                // If this joint is selected, wrap it with TransformControls:
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
                    // Not selected – just render as a sphere, no 3D axes
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

export default SMPLAnimated; 