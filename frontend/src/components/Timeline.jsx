import React, { useState, useEffect } from 'react';
import './Timeline.css';
import { BACKEND_URL } from './ModelViewer';
import * as THREE from 'three';
import NumberInput from "./NumberInput";

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
    onFrameChange,
    mode
}) {
    const [intervalId, setIntervalId] = useState(null);
    const [keyframes, setKeyframes] = useState([]);
    const [isUpdating, setIsUpdating] = useState(false);
    const [diffusionSteps, setDiffusionSteps] = useState(10);
    const [selectedModel, setSelectedModel] = useState('condMDI');

    const handleUpdateAnimation = async () => {
        if (jointPositions.length < 1) {
            console.log('No position history available');
            return;
        }
        setIsUpdating(true);
        try {
            // Transform original animation keyframes if they exist
            const originalKeyframes = jointPositions.allPositions ?
                jointPositions.allPositions : [];
            console.log("keyframes:", keyframes);
            console.log("oG,keys", originalKeyframes);

            const endpoint = selectedModel === 'priorMDM' ?
                'estimate_sequence' : 'generate_from_keyframes';

            const response = await fetch(`${BACKEND_URL}/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    originalKeyframes: originalKeyframes.map((kf, index) => ({
                        frame: index,
                        motionData: kf,
                    })),
                    keyframes: keyframes.map(kf => ({
                        frame: kf.frame,
                        motionData: kf.motionData,
                    })),
                    diffusion_steps: diffusionSteps
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

    const handleAddKeyframe = () => {
        setKeyframes(prev => {
            // Skip if no positions available
            if (!jointPositions?.current || !Array.isArray(jointPositions.current.positions)) return prev;

            const currentPositions = [...jointPositions.current.positions];
            const keyframeData = {
                frame: currentFrame,
                motionData: currentPositions
            }
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
        if (mode !== "editing") {
            const nextFrame = Math.min(currentFrame + 5, 195);
            onFrameChange(nextFrame);
        }
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
                    {mode === "editing" && (
                        <div className="select-container">
                            <label htmlFor="model-select">Model:</label>
                            <select
                                id="model-select"
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                            >
                                <option value="condMDI">condMDI</option>
                                <option value="priorMDM">priorMDM</option>
                            </select>
                        </div>
                    )}
                    <div className="select-container">
                        <label htmlFor="steps-select">Diffusion Steps:</label>
                        <select
                            id="steps-select"
                            value={diffusionSteps}
                            onChange={(e) => setDiffusionSteps(Number(e.target.value))}
                        >
                            <option value={10}>10</option>
                            <option value={100}>100</option>
                            <option value={1000}>1000</option>
                        </select>
                    </div>
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
