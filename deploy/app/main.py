import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from utils.preprocess import process_video_to_sequence, interpolate_keypoints_sequence, mediapipe_detection, extract_keypoints
import tensorflow as tf
import os, tempfile, json, uvicorn
import base64, cv2
import time
import asyncio
import numpy as np
import mediapipe as mp
from collections import deque
from fastapi.responses import HTMLResponse

N_HAND_LANDMARKS=21
N_UPPER_POSE_LANDMARKS=25
N_LANDMARKS=N_HAND_LANDMARKS*2 + N_UPPER_POSE_LANDMARKS

MODEL_PATH = "model/best_model_2011.keras"
LABEL_MAP = "model/label_map.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP, "r") as f:
    label_map = json.load(f)
label2text = {v: k for k, v in label_map.items()}

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

app = FastAPI(title="Sign Language Model API")

class SignResponse(BaseModel):
    label_int: int
    label_text: str
    confidence: float

# ---------------------------
# Root endpoint + HTML
# ---------------------------
# @app.get("/")
# def root():
#     return {"message": "Sign Language API is running"}
@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

# ---------------------------
# Offline video prediction
# ---------------------------
@app.post("/video-predict/", response_model=SignResponse)
async def predict_sign_from_video(video: UploadFile = File(...)):
    if not video.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="File must be a video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name
        
    keypoints_sequence = process_video_to_sequence(tmp_path, holistic)
    processed_frames = interpolate_keypoints_sequence(keypoints_sequence)
    
    X_input = np.array(processed_frames, dtype=np.float32)
    input_tensor = np.expand_dims(X_input, axis=0)
    
    preds = model.predict(input_tensor)
    label_int = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label_text = label2text.get(label_int, "Unknown")
    
    os.remove(tmp_path)
    
    return SignResponse(label_int=label_int, label_text=label_text, confidence=confidence)

# --------------------------------
# WebSocket realtime prediction
# --------------------------------
@app.websocket("/ws/stream-predict/")
async def stream_predict(websocket: WebSocket):
    await websocket.accept()
    
    keypoints_buffer = deque(maxlen=200)
    MAX_DURATION = 4.0
    start_time = None
    position = False
    hand_dection = False
    
    try:
        while True:
            # 1. Receive frame
            try:
                base64_frame = await websocket.receive_text()
            except Exception as e:
                print(f"ERROR: Fail to receive frame: {e}")
                break
            
            # 2. Decode frame
            nparr = np.frombuffer(base64.b64decode(base64_frame), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({
                    "status": "waiting",
                    "message": "Invalid frame received"
                })
                continue
            
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is None or len(keypoints) == 0:
                await websocket.send_json({
                    "status": "waiting",
                    "message": "No keypoints detected."
                })
                continue
            if not hand_dection:
                if np.all(keypoints[:N_HAND_LANDMARKS*2] == 0.0):
                    await websocket.send_json({
                        "status": "waiting",
                        "message": "No hand detected."
                    })
                    continue
                hand_dection = True
            
            if not position:
                await websocket.send_json({
                    "status": "position_ok",
                    "message": "OK! Hãy giữ nguyên tư thế. Bắt đầu ghi động tác..."
                })
                position = True
            
            if start_time is None:
                start_time = time.time()
                
            keypoints_buffer.append(keypoints)
            
            if len(keypoints_buffer) >= 30 and len(keypoints_buffer) % 30 == 0:
                if len(keypoints_buffer) > 30:
                    seq_array = np.array(keypoints_buffer[:60])  # shape: (len(keypoints_buffer), N_LANDMARKS, 3)
                seq_array = np.array(keypoints_buffer)
                # Interpolate
                processed_seq = interpolate_keypoints_sequence(seq_array) # shape: (30, N_LANDMARKS, 3)
                
                X_input = np.expand_dims(processed_seq, axis=0).astype(np.float32)
                preds = model.predict(X_input)
                
                label_int = int(np.argmax(preds))
                confidence = float(np.max(preds))
                label_text = label2text.get(label_int, "Unknown")
                
                # Gửi kết quả về client
                await websocket.send_json({
                    "label_int": label_int,
                    "label_text": label_text,
                    "confidence": confidence
                })
            
            # elapsed = time.time() - start_time
            # if elapsed > MAX_DURATION:
            #     await websocket.send_json({
            #         "status": "done",
            #         "message": "Max duration reached. Stop prediction."
            #     })
            #     keypoints_buffer.clear()
            #     start_time = None

    except WebSocketDisconnect:
        print("INFO: Client disconnected WebSocket.")
    except Exception as e:
        print(f"CRITICAL ERROR in WebSocket: {e}")
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason="Server error")
        except:
            pass
    finally:
        try: 
            await websocket.close()
        except:
            pass
        
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
