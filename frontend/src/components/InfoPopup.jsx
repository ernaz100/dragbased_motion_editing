import React from 'react';
import './InfoPopup.css';

function InfoPopup({ onClose }) {
    return (
        <div className="info-popup-overlay">
            <div className="info-popup">
                <button className="info-popup-close" onClick={onClose}>×</button>
                <h2>Overview</h2>
                <div className="info-section">
                    <p>
                        This project explores a new approach to character animation using a SMPL body model. By dragging joints to desired positions, users can easily pose and animate the 3D character, making the animation process more intuitive and accessible.
                    </p>
                </div>

                <div className="video-container">
                    <video
                        className="demo-video"
                        autoPlay
                        loop
                        muted
                        playsInline
                        style={{ playbackRate: 2.0 }}
                        onLoadedMetadata={(e) => { e.target.playbackRate = 2.0 }}
                    >
                        <source src="Motion_Editing.mov" type="video/mp4" />
                        Your browser does not support the video tag.
                    </video>
                </div>

                <div className="info-section">
                    <h3>Motion Synthesis</h3>
                    <p>
                        <strong>Selecting Joints:</strong> Click on any joint of the character to select it.
                        <br />
                        <strong>Dragging Joints:</strong> Once selected, you can drag a joint to a new position.
                        <br />
                        <strong>Updating Pose:</strong> After positioning joints, click "Update Pose" to have our model calculate a natural pose.
                    </p>
                </div>

                <div className="info-section">
                    <h3>Animation Mode</h3>
                    <p>
                        <strong>Adding Keyframes:</strong> Position your character and click "Add Keyframe" to mark important poses.
                        <br />
                        <strong>Generating Animation:</strong> After adding keyframes, click "Update Animation" to generate a smooth sequence where the diffusion model will try to find an animation to match the keyframes.
                        <br />
                        <strong>Playback:</strong> Use the timeline controls to play, pause, and scrub through your animation.
                    </p>
                </div>

                <div className="info-section">
                    <h3>Motion Editing</h3>
                    <p>
                        <strong>Editing Workflow:</strong> Select any frame in the animation timeline where you want to modify the pose.
                        <br />
                        <strong>Making Changes:</strong> Adjust the character's pose at your chosen frame by dragging joints. Don't forget to click Add Keyframe.
                        <br />
                        <strong>Diffusion Settings:</strong> Choose your diffusion model and number of steps (note: values above 10 may take longer to process).
                        <br />
                        <strong>Blending:</strong> The model will try to blend your modified pose with the original animation at that keyframe.
                    </p>
                </div>

                <div className="button-container">
                    <button className="start-button" onClick={onClose}>Get Started</button>
                </div>
            </div>
        </div>
    );
}

export default InfoPopup; 