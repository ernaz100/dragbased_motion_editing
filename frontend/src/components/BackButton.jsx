import React from 'react';
import './BackButton.css';

function BackButton({ onClick }) {
    return (
        <button className="back-button" onClick={onClick}>
            ← Back
        </button>
    );
}

export default BackButton; 