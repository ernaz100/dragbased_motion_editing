import React, { useEffect } from 'react';
import './LandingPage.css';

function LandingPage({ onModeSelect }) {
    useEffect(() => {
        const elements = document.querySelectorAll('h1, h2');
        let isDragging = false;
        let activeElement = null;
        let startX, startY;

        const handleMouseDown = (e) => {
            isDragging = true;
            activeElement = e.target;
            startX = e.clientX;
            startY = e.clientY;
        };

        const handleMouseMove = (e) => {
            if (!isDragging || !activeElement) return;

            const deltaX = (e.clientX - startX) * 0.1;
            const deltaY = (e.clientY - startY) * 0.1;

            activeElement.style.setProperty('--mouse-x', `${deltaX}px`);
            activeElement.style.setProperty('--mouse-y', `${deltaY}px`);
        };

        const handleMouseUp = () => {
            if (activeElement) {
                activeElement.style.setProperty('--mouse-x', '0px');
                activeElement.style.setProperty('--mouse-y', '0px');
            }
            isDragging = false;
            activeElement = null;
        };

        elements.forEach(element => {
            element.addEventListener('mousedown', handleMouseDown);
        });
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            elements.forEach(element => {
                element.removeEventListener('mousedown', handleMouseDown);
            });
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    return (
        <div className="landing-page">
            <h1 className='title'>Exploring Motion Editing Methods with Drag-Based Signals</h1>
            {        //    <h2>A Virtual Humans project by Eric Nazarenus, Patricia Schlegel and Daniel Flat</h2>
            }            <div className="button-container">
                <button
                    className="mode-button"
                    onClick={() => onModeSelect('synthesis')}
                >
                    Motion Synthesis
                </button>
                <button
                    className="mode-button"
                    onClick={() => onModeSelect('editing')}
                >
                    Motion Editing
                </button>
            </div>
        </div>
    );
}

export default LandingPage; 