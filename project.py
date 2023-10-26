import cv2
import numpy as np
import PoseModule as pm
from flask import Flask, render_template, request, redirect,Response,url_for
from werkzeug.utils import secure_filename
import os
import cv2
import time
import json
from PIL import Image
from io import BytesIO
import json
import numpy as np
from datetime import timedelta

app = Flask(__name__)
detector = pm.poseDetector()
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.form.get("name")
        print("第一个f")
        print(f)
        if len(f)==0:  # 上传视频文件
            g = request.files['file']
            f = g.filename
            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            print(basepath)
            upload_path = os.path.join(basepath,secure_filename(g.filename))
            print(upload_path)
            upload_path = os.path.abspath(upload_path) # 将路径转换为绝对路径
            print(upload_path)
            g.save(upload_path)
            f = upload_path
            print(f)
        # local Camera
        if len(f) == 1:
            f = 0
        # RTSP

        return redirect(url_for('video_feed',path = f))
    return render_template('upload.html')


def gen_frames(path):  # generate frame by frame from camera
    count = 0
    dir = 0
    pTime = 0
    print("第二个f")
    print(path)
    # if path[0]!='r':
    #     path='/'+path
    print(type(path))
    if path == '0':
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(path)
    #camera = cv2.VideoCapture("8.mp4")
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (960, 576))
            frame = detector.findPose(frame, False)
            lmList = detector.findPosition(frame, False)
            # print(lmList)
            if len(lmList) != 0:
                # Right Arm
                #angle = detector.findAngle(frame, 12, 24, 26)
                angle = detector.findAngle(frame, 12, 14, 16)
                # # Left Arm
                #angle = detector.findAngle(frame, 11, 13, 15)
                # per = np.interp(angle, (210, 310), (0, 100))
                per = np.interp(angle, (30, 130), (0, 100))
                bar = np.interp(angle, (30, 130), (100, 550))
                # bar = np.interp(angle, (210, 310), (650, 100))

                # Check for the dumbbell curls
                color = (255, 0, 255)
                if per >= 80:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per <= 40:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0
                print(per, count)

                # Draw Bar
                cv2.rectangle(frame, (800, 100), (875, 550), color, 3)
                cv2.rectangle(frame, (800, int(bar)), (875, 550), color, cv2.FILLED)
                cv2.putText(frame, f'{int(100 - per)} %', (800, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                            color, 4)

                # Draw Curl Count
                cv2.rectangle(frame, (0, 376), (250, 576), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, str(int(count)), (30, 550), cv2.FONT_HERSHEY_PLAIN, 10,
                            (255, 0, 0), 25)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.rectangle(frame, (0, 0), (200, 150), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 8,
                        (255, 0, 0), 15)


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed/<path:path>')
def video_feed(path):
    print(path)
    return Response(gen_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
