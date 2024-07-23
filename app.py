from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import cv2
import pytesseract

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_text_regions(frame):
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    
    # Prepare the frame to be fed to the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    
    # Get the output layers where the predictions are stored
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    scores, geometry = net.forward(output_layers)
    
    # Decode the predictions to get text regions
    conf_threshold = 0.5
    nms_threshold = 0.4
    rects = []
    confidences = []
    
    num_rows, num_cols = scores.shape[2:4]
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        
        for x in range(num_cols):
            score = scores_data[x]
            if score < conf_threshold:
                continue
                
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles_data[x]
            cos_a, sin_a = cos(angle), sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            
            end_x = int(offset_x + cos_a * x1_data[x] + sin_a * x2_data[x])
            end_y = int(offset_y - sin_a * x1_data[x] + cos_a * x2_data[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(score))
    
    # Apply Non-Max Suppression to filter out overlapping boxes
    boxes = cv2.dnn.NMSBoxesRotated(rects, confidences, conf_threshold, nms_threshold)
    return boxes

def extract_text_from_video(video_path):
    frames = extract_frames(video_path)
    extracted_text = []
    
    for frame in frames:
        boxes = detect_text_regions(frame)
        
        for (start_x, start_y, end_x, end_y) in boxes:
            roi = frame[start_y:end_y, start_x:end_x]
            text = pytesseract.image_to_string(roi, config='--psm 6')
            extracted_text.append(text)
    
    return " ".join(extracted_text)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            extracted_text = extract_text_from_video(video_path)
            return render_template('result.html', text=extracted_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)