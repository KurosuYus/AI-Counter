import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy import linalg

# 视频设备号
#DEVICE_NUM = 0
DEVICE_NUM = "hand1.mp4"


# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3):
    result = 0
    # 计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1

    return result
#使用聚拢思路，构建包点，识别出7
def lseven_cross(ls1,ls2,ls3):
    cross = 0  #默认不聚拢
    l1 = np.linalg.norm((ls2 - ls1), ord=2)
    l2 = np.linalg.norm((ls3 - ls1), ord=2)
    if (l1 > 30) and (l1 < 70) and (l2 > 30) and (l2 < 70):
        cross = 1

    return cross
#同理前1-6的算法，识别出9
def lnine_stretch_detect(ln1,ln2,ln3):
    lin = 0
    n1 = np.linalg.norm((ln2 - ln1), ord=2)
    n2 = np.linalg.norm((ln3 - ln1), ord=2)
    if (n1 > n2):
        lin = 1

    return lin
# 检测手势
def detect_hands_gesture(result,cross,lin):
    if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "good",0
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "1",1
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "please civilization in testing", -1
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "2",2
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
        gesture = "3",3
    elif (result[0] == 0) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "four",4
    elif (result[0] == 1) and (result[1] == 1) and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "five",5
    elif (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
        gesture = "six",6
    elif (cross == 1):
        gesture = "seven",7
    elif (result[0] == 1) and (result[1] == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "eight",8
    elif (result[0] == 0) and (lin == 1) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "nine",9
    elif (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
        gesture = "ten",10
    else:
        gesture = "not in detect range...",-11
    return gesture




def detect():
    # 接入USB摄像头时，注意修改cap设备的编号
    cap = cv.VideoCapture(DEVICE_NUM)
    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

    figure = np.zeros(5)
    landmark = np.empty((21, 2))

    if not cap.isOpened():
        print("Can not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break

        # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # print(result.multi_hand_landmarks)
        # 如果检测到手
        if result.multi_hand_landmarks:
            # 为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame,
                                      handLms,
                                      mpHands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=handLmsStyle,
                                      connection_drawing_spec=handConStyle)

                for j, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * frame_width)
                    yPos = int(lm.y * frame_height)
                    landmark_ = [xPos, yPos]
                    landmark[j, :] = landmark_

                # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
                for k in range(5):
                    if k == 0:
                        figure_ = finger_stretch_detect(landmark[17], landmark[4 * k + 2], landmark[4 * k + 4])
                    else:
                        figure_ = finger_stretch_detect(landmark[0], landmark[4 * k + 2], landmark[4 * k + 4])
                        cross = lseven_cross(landmark[4], landmark[8], landmark[12])
                        lin = lnine_stretch_detect(landmark[0], landmark[7], landmark[8])

                    figure[k] = figure_
                print(figure, '\n')

                gesture_result,_ = detect_hands_gesture(figure,cross,lin)
                cv.putText(frame, f"{gesture_result}", (30, 60 * (i + 1)), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 5)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



def detectfigure(camera):

    # 接入USB摄像头时，注意修改cap设备的编号
    #cap = cv.VideoCapture(0)
    cap = camera

    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))

    figure = np.zeros(5)
    landmark = np.empty((21, 2))

    if not cap.isOpened():
        print("Can not open camera.")
        exit()

    gesture_result = None
    gesture_num = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break

        # mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        # 读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # print(result.multi_hand_landmarks)
        # 如果检测到手
        if result.multi_hand_landmarks:
            # 为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame,
                                      handLms,
                                      mpHands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=handLmsStyle,
                                      connection_drawing_spec=handConStyle)

                for j, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * frame_width)
                    yPos = int(lm.y * frame_height)
                    landmark_ = [xPos, yPos]
                    landmark[j, :] = landmark_

                # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
                for k in range(5):
                    if k == 0:
                        figure_ = finger_stretch_detect(landmark[17], landmark[4 * k + 2], landmark[4 * k + 4])
                    else:
                        figure_ = finger_stretch_detect(landmark[0], landmark[4 * k + 2], landmark[4 * k + 4])
                        cross = lseven_cross(landmark[4], landmark[8], landmark[12])
                        lin = lnine_stretch_detect(landmark[0], landmark[7], landmark[8])

                    figure[k] = figure_
                #print(figure, '\n')

                gesture_result, gesture_num = detect_hands_gesture(figure,cross,lin)

                cv.putText(frame, f"{gesture_result}", (30, 60 * (i + 1)), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 5)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            gesture_num = 0
            break

        gestures = [1,2,3,4,5,6,7,8,9,10,11,0]

        #print("reco:", gesture_num)
        if gesture_num in gestures:
            #print("break:", gesture_num)
            break
    #cap.release()
    #cv.destroyAllWindows()
    return gesture_num


def beginOK(camera, figurelist):

    cnt = 0

    while True:

        res = detectfigure(camera)
        if res == 11 and cnt == 0:
            cnt += 1
        if res in figurelist and cnt == 1:
            cnt += 1
        if cnt == 2:
            return res
            

if __name__ == '__main__':
    #detect()
    camera =  cv.VideoCapture(0)
    print(detectfigure(camera))