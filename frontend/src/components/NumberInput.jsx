import React from 'react';

function NumberInput({value, onChangeFunction}) {

  const handleChange = (e) => {
    const newValue = e.target.value;
    if (!isNaN(newValue)) {
      onChangeFunction(newValue);
    }
  };

  return (
    <div>
        <span>Number of Diffusion Steps</span>
        <input
        type="number"
        value={value}
        onChange={handleChange}/>
    </div>
  );
}

export default NumberInput;