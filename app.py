import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from flask import Flask, request, jsonify, send_file
import io
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMG_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VID_EXTENSIONS = {'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

json_file = open('utils/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
model.load_weights('utils/fer.h5')
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def detecte_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    check = True
    if len(faces_detected) > 0:
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            # print("roigray shape: ",roi_gray.shape)
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0
        return img_pixels, x, y, check
    else:
        # cv2.putText(img, 'No faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(frame, 'No faces detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        roi_gray = cv2.resize(gray_img, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        check = False
        return img_pixels, 25, 25, check

@app.route('/vid_emotion', methods=['GET', 'POST'])
def vid_emotion():
    if 'video' not in request.files:
        return "No file part"
    file = request.files['video']
    if file.filename == '':
        return "No selected file"
    file.save(f"{app.config['UPLOAD_FOLDER']}/videos/{file.filename}")
    cap = cv2.VideoCapture(f"{app.config['UPLOAD_FOLDER']}/videos/{file.filename}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    print('current fps: ',fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f"{app.config['UPLOAD_FOLDER']}/videos/results/{file.filename}", fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret == True:
            img_pixels, x, y, check = detecte_face(frame)
            if check == True:
                predictions = model.predict(img_pixels)
                max_index = int(np.argmax(predictions))
                predicted_emotion = emotions[max_index]
            else:
                predicted_emotion = "No face found"

            
            print(predicted_emotion)
            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow('frame', frame)
            output.write(frame)
        else:
            break
    # Release video capture and writer
    cap.release()
    output.release()
    video_path = f"{app.config['UPLOAD_FOLDER']}/videos/results/{file.filename}"
    return send_file(video_path, mimetype='video/mp4', as_attachment=True)

@app.route('/img_emotion', methods=['GET', 'POST'])
def img_emotion():
    if 'frame' not in request.files:
        return "No file part"
    file = request.files['frame']

    if file.filename == '':
        return "No selected file"
    #store image
    file.save(f"{app.config['UPLOAD_FOLDER']}/images/{file.filename}")

    img = cv2.imread(f"{app.config['UPLOAD_FOLDER']}/images/{file.filename}", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))
    img_pixels, x, y, check = detecte_face(img)
    if check == True:
        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))
        predicted_emotion = emotions[max_index]
    else:
        predicted_emotion = "No face found"

    
    print(predicted_emotion)
    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imwrite(f"{app.config['UPLOAD_FOLDER']}/images/results/{file.filename}",img)
    # cv2.imshow('Facial Emotion Recognition', resized_img)

    return jsonify({'emotion':predicted_emotion})

@app.route('/check', methods =['GET','POST'])
def check():
    return jsonify({'status': 200, 'message': "Welcome to AWS EC2"})

@app.route('/', methods=['GET', 'POST'])
def hello_world():
	return 'Hello World!!?'

if __name__=='__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8000)
