import React, { useState, useRef } from 'react';
import EditingViewer from './components/EditingViewer';
import SynthesisViewer from './components/SynthesisViewer';
import Timeline from './components/Timeline';
import LandingPage from './components/LandingPage';
import './App.css';

function App() {
    // Add mode state
    const [mode, setMode] = useState(null);

    // Current animation time in seconds
    const [currentTime, setCurrentTime] = useState(0);

    const [sequencePositions, setSequencePositions] = useState([]);

    // Store whether or not the animation is playing
    const [isPlaying, setIsPlaying] = useState(false);

    const [animationInfo, setAnimationInfo] = useState({
        duration: 100,
        numKeyframes: 0,
        framesPerSecond: 20
    });

    // Add ref to store the updatePose function from ModelViewer
    const updatePoseRef = useRef(null);

    // Add separate states for different types of position data
    const [allPositions, setAllPositions] = useState([]);
    const [currentPositions, setCurrentPositions] = useState(null);

    // Add currentFrame state at App level
    const [currentFrame, setCurrentFrame] = useState(0);

    const handleUpdatePose = () => {
        if (updatePoseRef.current) {
            updatePoseRef.current();
        } else {
            console.log('App: updatePoseRef.current is null');
        }
    };

    const handleBack = () => {
        setMode(null);
    };

    // If no mode is selected, show landing page
    if (!mode) {
        return <LandingPage onModeSelect={setMode} />;
    }

    return (
        <div className="app">
            <div className="viewer-container">
                {mode === 'editing' ? (
                    <EditingViewer
                        currentTime={currentTime}
                        onAnimationLoaded={setAnimationInfo}
                        onUpdatePoseRef={(fn) => updatePoseRef.current = fn}
                        setCurrentJointPositionCallback={setCurrentPositions}
                        setAllPositions={setAllPositions}
                        sequencePositions={sequencePositions}
                        isPlaying={isPlaying}
                        onBack={handleBack}
                        currentFrame={currentFrame - 1}
                    />
                ) : (
                    <SynthesisViewer
                        onBack={handleBack}
                        onUpdatePoseRef={(fn) => updatePoseRef.current = fn}
                        setAnimationInfo={setAnimationInfo}
                        setCurrentJointPositionCallback={setCurrentPositions}
                        sequencePositions={sequencePositions}
                        currentFrame={currentFrame - 1}
                    />
                )}
            </div>
            <div className="timeline-container">
                <Timeline
                    onTimeChange={(timeInSeconds) => setCurrentTime(timeInSeconds)}
                    totalDuration={animationInfo.duration}
                    numKeyframes={animationInfo.numKeyframes}
                    framesPerSecond={animationInfo.framesPerSecond}
                    isPlaying={isPlaying}
                    setIsPlaying={setIsPlaying}
                    onUpdatePose={handleUpdatePose}
                    jointPositions={{
                        allPositions: allPositions,
                        current: currentPositions
                    }}
                    setSequencePositions={setSequencePositions}
                    onFrameChange={setCurrentFrame}
                    currentFrame={currentFrame}
                />
            </div>
        </div>
    );
}

export default App; 