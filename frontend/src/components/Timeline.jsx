import React, { useState, useEffect } from 'react';
import './Timeline.css';
import { BACKEND_URL } from './ModelViewer';
import * as THREE from 'three';

function Timeline({
    onTimeChange,
    totalDuration,
    numKeyframes,
    framesPerSecond,
    isPlaying,
    setIsPlaying,
    onUpdatePose,
    jointPositions,
    setSequencePositions,
    currentFrame,
    onFrameChange
}) {
    const [intervalId, setIntervalId] = useState(null);
    const [keyframes, setKeyframes] = useState([]);
    const [isUpdating, setIsUpdating] = useState(false);

    const handleUpdateAnimation = async () => {
        if (jointPositions.length < 1) {
            console.log('No position history available');
            return;
        }
        setIsUpdating(true);
        try {
            // Transform original animation keyframes if they exist
            const originalKeyframes = jointPositions.allPositions ?
                jointPositions.allPositions.map((positions, frameIndex) =>
                    transformJointsToKeyframeData(
                        positions,
                        frameIndex,
                        framesPerSecond,
                        frameIndex > 0 ? { positions: jointPositions.allPositions[frameIndex - 1] } : null
                    )
                ) : [];
            console.log(originalKeyframes);
            console.log(keyframes);
            const response = await fetch(`${BACKEND_URL}/generate_from_keyframes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    originalKeyframes: originalKeyframes.map(kf => ({
                        frame: kf.frame,
                        motionData: kf.motionData,
                    })),
                    keyframes: keyframes.map(kf => ({
                        frame: kf.frame,
                        motionData: kf.motionData,
                    }))
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const { generated_motion } = data;

            // Convert the generated_motion array into frame-by-frame positions
            // Each frame will contain positions for all joints
            const framePositions = Array(196).fill(0).map((_, frameIdx) => {
                // For each frame, collect all joint positions
                return generated_motion.map((joint, jointIdx) => {
                    // Each joint has x, y, z arrays
                    return [
                        joint[0][frameIdx], // x
                        joint[1][frameIdx], // y
                        joint[2][frameIdx]  // z
                    ];
                });
            });

            // Update the sequence positions with the new animation
            setSequencePositions(framePositions);

            // Reset to beginning of animation
            onFrameChange(0);

        } catch (error) {
            console.error('Error updating sequence:', error);
        } finally {
            setIsUpdating(false);
        }
    };

    const handleSliderChange = (value) => {
        const newFrame = parseInt(value);
        onFrameChange(newFrame);
    };

    const transformJointsToKeyframeData = (currentPositions, currentFrame, framesPerSecond, prevKeyframe = null) => {
        const rootPosition = currentPositions[1];

        // Filter out joints 0, 24, and 19 (root and hands)
        const filteredPositions = currentPositions.filter((_, index) =>
            ![0, 24, 19].includes(index)
        );

        // x_p_t: Local joint positions (22 × 3)
        const localJointPositions = filteredPositions.map(pos => [
            pos[0] - rootPosition[0],
            pos[1],
            pos[2] - rootPosition[2]
        ]);
        console.log("local", localJointPositions);

        // Extract joint rotations from the SMPL model
        // Each joint rotation is represented in 6D rotation format
        const jointRotations = Array(21).fill([1, 0, 0, 0, 1, 0]); // Default identity rotation if no rotations available
        console.log("jr", jointRotations);

        // x_global_t: Global motion (4 values total)
        let currentDeltas = {
            rootRotationDelta: 0,
            rootPositionDelta: [rootPosition[0], rootPosition[2]],
            rootHeight: rootPosition[1]
        };
        let globalMotion = {
            rootRotationDelta: currentDeltas.rootRotationDelta,
            rootPositionDelta: [...currentDeltas.rootPositionDelta],
            rootHeight: currentDeltas.rootHeight
        };

        // ẋ_p_t: Joint velocities (22 joints × 3 dimensions = 66 values)
        const jointVelocities = filteredPositions.map((pos, idx) => {
            if (!prevKeyframe) return [0, 0, 0];
            const prevPos = prevKeyframe.positions[idx];
            const frameTime = 1 / framesPerSecond;
            return [
                (pos[0] - prevPos[0]) / frameTime,
                (pos[1] - prevPos[1]) / frameTime,
                (pos[2] - prevPos[2]) / frameTime
            ];
        });
        console.log("jv", jointVelocities);

        // c_t: Foot contact (4 values, one for each foot vertex)
        const footContact = [
            currentPositions[8][1] < 0.1 ? 1 : 0,  // left_heel (joint 8)
            currentPositions[9][1] < 0.1 ? 1 : 0,  // left_toe (joint 9)
            currentPositions[4][1] < 0.1 ? 1 : 0,  // right_heel (joint 4)
            currentPositions[5][1] < 0.1 ? 1 : 0   // right_toe (joint 5)
        ];
        // 
        return {
            frame: currentFrame,
            positions: filteredPositions,
            motionData: {
                // x_global_t ∈ R⁴
                global: globalMotion,
                // x_local_t ∈ R²⁵⁹
                local: {
                    jointPositions: localJointPositions.slice(1),//    R⁶³  (21x3)
                    jointRotations: jointRotations,         // R¹²⁶ (21×6)
                    jointVelocities: jointVelocities,       // R⁶⁶ (22×3)
                    footContact: footContact                // R⁴
                }
            }
        };
    };

    const handleAddKeyframe = () => {
        setKeyframes(prev => {
            // Skip if no positions available
            if (!jointPositions?.current || !Array.isArray(jointPositions.current.positions)) return prev;

            const currentPositions = [...jointPositions.current.positions];
            console.log("curr", currentPositions);

            const prevKeyframe = prev.length > 0 ? prev[prev.length - 1] : null;

            const keyframeData = transformJointsToKeyframeData(
                currentPositions,
                currentFrame,
                framesPerSecond,
                prevKeyframe
            );

            // Check if frame already exists
            const existingIndex = prev.findIndex(kf => kf.frame === currentFrame);
            if (existingIndex >= 0) {
                // Replace existing keyframe
                const newKeyframes = [...prev];
                newKeyframes[existingIndex] = keyframeData;
                return newKeyframes;
            } else {
                // Add new keyframe
                return [...prev, keyframeData].sort((a, b) => a.frame - b.frame);
            }
        });

        // After adding the keyframe, move the current frame forward by 20
        const nextFrame = Math.min(currentFrame + 20, 195);
        onFrameChange(nextFrame);
    };

    // Start/stop autoplay
    const handlePlayToggle = () => {
        if (!isPlaying) {
            setIsPlaying(true);

            const newIntervalId = setInterval(() => {
                onFrameChange(prevFrame => {
                    if (prevFrame >= numKeyframes) {
                        clearInterval(newIntervalId);
                        setIsPlaying(false);
                        return numKeyframes;
                    }
                    return prevFrame + 1;
                });
            }, framesPerSecond === 0 ? 1000 / 20 : 1000 / framesPerSecond);  // Convert FPS to milliseconds interval
            setIntervalId(newIntervalId);
        } else {
            setIsPlaying(false);
            clearInterval(intervalId);
        }
    };

    const handleKeyframeClick = (frame) => {
        onFrameChange(frame);
    };

    useEffect(() => {
        const newTime = (currentFrame / framesPerSecond).toFixed(2);
        onTimeChange(newTime);
    }, [currentFrame, totalDuration]);

    // Add this helper function to check if there are keyframes after frame 40
    const hasValidKeyframes = () => {
        return keyframes.some(kf => kf.frame >= 40);
    };

    return (
        <div className="timeline">
            <div className="timeline-controls">
                <div>
                    <button onClick={handlePlayToggle}>
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button onClick={() => {
                        onUpdatePose();
                    }}>
                        Update Pose
                    </button>
                    <button
                        onClick={handleUpdateAnimation}
                        disabled={!hasValidKeyframes() || isUpdating}
                        title={!hasValidKeyframes() ? "Add at least one keyframe after frame 40 to enable animation update" : ""}
                        className={!hasValidKeyframes() ? "button-disabled" : ""}
                    >
                        {isUpdating ? "Updating..." : "Update Animation"}
                    </button>
                    <button onClick={handleAddKeyframe}>Add Keyframe</button>
                </div>
                <button onClick={() => {
                    setSequencePositions(null);
                    setKeyframes([]);
                    onFrameChange(0);
                }}>Restart</button>
            </div>
            <div className="timeline-slider">
                <div className="keyframes">
                    {keyframes.map(({ frame }) => (
                        <div
                            key={frame}
                            className="keyframe-marker"
                            style={{ left: `${(frame / numKeyframes) * 100}%` }}
                            data-frame={`Frame ${frame}`}
                            onClick={() => handleKeyframeClick(frame)}
                        />
                    ))}
                    <input
                        type="range"
                        min="0"
                        max={numKeyframes}
                        step={1}
                        value={currentFrame}
                        onChange={(e) => handleSliderChange(e.target.value)}
                    />
                </div>
                <div className="animation-info">
                    Keyframe: {currentFrame} | FPS: {framesPerSecond} |
                    Time: {framesPerSecond === 0 ? '0' : (currentFrame / framesPerSecond).toFixed(2)}s
                </div>
            </div>
        </div>
    );
}

export default Timeline;
