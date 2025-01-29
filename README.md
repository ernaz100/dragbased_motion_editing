# SMPL Model Viewer with Pose Estimation

A web application for viewing and manipulating SMPL 3D human models with real-time pose estimation. The project consists of a Flask backend for pose estimation and a React frontend for model visualization.

## Prerequisites

- Python 3.8+ (for backend)
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
1. Download the model files from our Google Drive: https://drive.google.com/drive/folders/18UGjxcR_ii4SNqyS8O0j-3A4wWwy00Bn?usp=drive_link and place into the project folder

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
pip install -r requirements.txt
```

5. Create a `static` directory for storing generated files:
```bash
mkdir static
```

6. Start the Flask server:
```bash
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
