import React from 'react';
import "./NumberInput.css"

function NumberInput({value, onChangeFunction}) {

  const handleChange = (e) => {
    const newValue = e.target.value;
    if (!isNaN(newValue)) {
      onChangeFunction(newValue);
    }
  };

  return (
    <div className="number-input-container">
        <div>Number of Diffusion Steps</div>
        <input
        type="number" className="number-input-field"
        value={value}
        onChange={handleChange}/>
    </div>
  );
}

export default NumberInput;