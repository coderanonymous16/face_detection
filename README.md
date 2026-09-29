# Face Detection App

A face detection project built with **Python, OpenCV (Haar Cascade)** and a **Streamlit** frontend.
It started as four separate scripts and is now one app with three modes: image, camera and video.

## Features

- Detect faces in an uploaded image (single or multiple faces)
- Real-time face detection from the webcam (Live tab, via `streamlit-webrtc`)
- Capture a photo from the camera and detect faces
- Upload a video and preview detections on sampled frames
- Adjustable detection settings (scale factor, min neighbors)
- Random-coloured bounding boxes with a face count

## Tech Stack

| Part      | Tool                              |
|-----------|-----------------------------------|
| Language  | Python 3.9+                       |
| Detection | OpenCV Haar Cascade               |
| Frontend  | Streamlit                         |
| Deploy    | Streamlit Community Cloud (free)  |

## Project Structure

```
face_detection_app/
├── app.py              # unified app (detection logic + UI)
├── requirements.txt    # dependencies
├── README.md
└── legacy/             # original scripts (optional, for local use)
    ├── FACEDETECTION_only1.py
    ├── FACEDETECTION_multi.py
    ├── FACEDETECTION_webcam.py
    └── tempCodeRunnerFile.py
```

## How the Old Scripts Map to the App

| Old script                | What it did            | Now in `app.py`                              |
|---------------------------|------------------------|----------------------------------------------|
| `FACEDETECTION_only1.py`  | first face only        | Sidebar: "Only first face"                   |
| `FACEDETECTION_multi.py`  | all faces in an image  | Sidebar: "All faces" + Image tab             |
| `FACEDETECTION_webcam.py` | live webcam loop       | Live tab (real time) + Camera tab (snapshot) |
| `tempCodeRunnerFile.py`   | grayscale test         | merged into `detect_faces()`                 |

The common part of all four scripts (load cascade, convert to grayscale, `detectMultiScale`, draw rectangles) is now a single function, `detect_faces()`. Every tab calls it, so a fix in one place applies everywhere.

## Run Locally

```bash
# 1. go to the project folder
cd face_detection_app

# 2. (optional) virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> Note: `facedata.xml` is not needed. The app loads the cascade that ships with OpenCV.

## Deploy (Streamlit Community Cloud, free)

1. **Push to GitHub**
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "face detection app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/face-detection-app.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app**, choose your repo, branch `main`, and main file `app.py`.
4. Click **Deploy**. In a minute or two you get a public link like `https://<app-name>.streamlit.app`.

Add that link at the top of this README once it is live.

### Deployment notes

- Use `opencv-python-headless` (already in `requirements.txt`). Plain `opencv-python` often fails on servers because of missing GUI libraries.
- Keep OpenCV on version 4 (`opencv-python-headless<5`). In OpenCV 5 the Haar `CascadeClassifier` moved to the contrib module, so `cv2.CascadeClassifier` is missing there.
- `cv2.imshow` and `cv2.VideoCapture(0)` do not work on a server, because there is no screen and no server-side camera. That is why the app uses the browser camera (`st.camera_input`) instead.
- Do not commit large videos or images to the repo. Keep sample files small.

### Alternatives

- **Hugging Face Spaces**: create a Streamlit Space and upload the same files.
- **Render / Railway**: use the start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Limitations

- Haar Cascade works best on frontal, well-lit faces. Side profiles, masks and small faces may be missed.
- The video tab previews sampled frames instead of processing the full video.
- The Live tab may lag on slow machines, and on some networks (college or office Wi-Fi) the WebRTC connection can fail without a TURN server.

## Future Improvements

- Better accuracy with a DNN detector (OpenCV DNN, MediaPipe or YuNet)
- Full video processing with a downloadable output
- Face recognition (who is this person) on top of detection

## License

Add a license of your choice (for example MIT).
