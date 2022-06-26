from flask import Flask, render_template, Response
import cv2

from utils_tf import predict_image
import tensorflow as tf

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
classes = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

model = tf.keras.models.load_model("../model/SGD_lr_1e-05_kernel_size_3.h5")
#
# stage, percentage = predict_image(frame, model)
# print(stage)
# print(percentage)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            stage, accuracy = predict_image(frame, model)
            print(stage)
            print(accuracy)
            cv2.putText(img=frame, text=f"{classes[stage]}, Accuracy: {accuracy*100}", org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(0, 0, 0), thickness=2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    # camera = cv2.VideoCapture(0)  # use 0 for web camera
    # #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
    # # for local webcam use cv2.VideoCapture(0)
    #
    # model = tf.keras.models.load_model("model/first_tensorflow_model.h5")
    # img = cv2.imread('../dataset/stage_1/IMG_5032.jpg')
    # stage, percentage = predict_image(img, model)
    # print(stage)
    # print(percentage)