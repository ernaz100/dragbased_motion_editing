# SMPL Model Viewer with Pose Estimation

A web application for viewing and manipulating SMPL 3D human models with real-time pose estimation. The project consists of a Flask backend for pose estimation and a React frontend for model visualization.

## Prerequisites

- Python 3.8.10 (for backend)
- Node.js 14+ (for frontend)
- Git

## Project Structure

```
.
├── backend/         # Flask server
├── frontend/        # React application
├── models/smpl/       # SMPL Model files
└── README.md
```

## Download Model files

1. Download the pose network weights from our Google Drive: https://drive.google.com/file/d/1kEuWw1R6AgT36nEVBch9Zkq6u9oLVWxE/view?usp=sharing and place it into backend/checkpoints
2. Download the SMPL Body Model https://drive.google.com/file/d/1WJyEHeKGddPo8DSvYfOwBHjGAjwlUTfn/view?usp=drive_link and place it into models/ (you want this structure: models/smpl/SMPL_FEMALE.pkl )
3. Download the priorMDM left wrist finetuned model https://drive.google.com/file/d/17h98FQhu6dFj70YCopFHT4sL6jZOf42U/view , unzip and place the model000280000.pt file into backend/priorMDM/save/left_wrist_finetuned/
4. Download the condMDI model: https://drive.google.com/file/d/1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j/view, unzip and place the model000750000.pt file into backend/diffusion_motion_inbetweening/save/condmdi_randomframes
5. Download the HumanML3D dataset: and place it into backend/diffusion_motion_inbetweening/dataset/HumanML3D

## Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```


4. Install the required Python packages:
```bash
pip install wheel
pip install setuptools
pip install chumpy --no-build-isolation
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

5. Because of dependency issues you need to go into the installed chumpy package and delete the line 11 from __init__.py (e.g line 11 at /.pyenv/versions/3.8.10/lib/python3.8/site-packages/chumpy/__init__.py):  
```bash
from numpy import bool, int, float, complex, object, unicode, str, nan, inf ## delete this
```

6. Download the necesssary data for condMDI:
```bash
cd diffusion_motion_inbetweening/
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_smpl_files.sh
bash prepare/download_recognition_unconstrained_models.sh
```


7. Start the Flask server:
```bash
cd ..
python app.py
```

The backend server will start running on `http://localhost:5001`

## Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

2. Install the required Node.js packages:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend application will start running on `http://localhost:3000`

## Using the Application

1. Open your web browser and navigate to `http://localhost:3000`
2. The application will display a 3D human model that you can:
   - Rotate and zoom using mouse controls
   - Select joints by clicking on them
   - Drag joints to modify poses
   - Use the timeline controls to play animations
   - Update poses using the "Update Pose" button
   - Update animations by changing poses and adding the keyframes, then pressing Update Animation

## Development

- Backend code is in Python using Flask and the SMPL model for pose estimation
- Frontend is built with React and Three.js for 3D visualization
- The backend serves static files (like GLB models) from the `backend/static` directory
- CORS is enabled for development between frontend and backend servers

## Troubleshooting

1. If you see CORS errors:
   - Ensure both servers are running
   - Check that the backend URL in the frontend code matches your setup

2. If the model doesn't load:
   - Check the browser console for errors
   - Ensure the GLB file paths are correct
   - Verify that the static directory exists in the backend

3. If pose estimation fails:
   - Check the backend logs for detailed error messages
   - Verify that all required Python packages are installed correctly
