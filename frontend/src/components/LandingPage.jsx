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

        // Add star background
        const starsContainer = document.createElement('div');
        starsContainer.className = 'stars';
        document.querySelector('.landing-page').appendChild(starsContainer);

        // Generate random stars
        const numberOfStars = 100;
        for (let i = 0; i < numberOfStars; i++) {
            const star = document.createElement('div');
            star.className = 'star';

            // Random position
            star.style.left = `${Math.random() * 100}%`;
            star.style.top = `${Math.random() * 100}%`;

            // Random size
            const size = Math.random() * 3 + 1;
            star.style.width = `${size}px`;
            star.style.height = `${size}px`;

            // Random twinkle duration
            star.style.setProperty('--twinkle-duration', `${Math.random() * 3 + 2}s`);

            starsContainer.appendChild(star);
        }

        return () => {
            elements.forEach(element => {
                element.removeEventListener('mousedown', handleMouseDown);
            });
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
            starsContainer.remove();
        };
    }, []);

    return (
        <div className="landing-page">
            <h1 className='title'>Exploring Motion Editing Methods with Drag-Based Signals</h1>
            {<h2>A Virtual Humans project by Eric Nazarenus, Patricia Schlegel and Daniel Flat</h2>
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