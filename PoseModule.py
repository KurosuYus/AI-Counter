# -*- codeing =utf-8 -*-
import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self,mode=False,upBody=False,smooth=True,
                 detectionCon=0.5,trackCon=0.5):
        '''

        :param mode:是否是静态图片，默认为否
        :param upBody:是否是上半身，默认为否
        :param smooth:设置为True减少抖动
        :param detectionCon:人员检测模型的最小置信度值，默认为0.5
        :param trackCon::姿势可信标记的最小置信度值，默认为0.5
        '''



        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # 创建一个Pose对象用于检测人体姿势
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,False,self.detectionCon,self.trackCon)
        #self.pose = mp.solutions.pose.Pose(self.static_image_mode, self.upper_body_only, self.smooth_landmarks,False,self.min_detection_confidence, self.min_tracking_confidence)
    # 检测人体姿势
    def findPose(self,img,draw=True):
        '''
     检测姿势方法
              :param img: 一帧图像
              :param draw: 是否画出人体姿势节点和连接图
              :return: 处理过的图像
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) #会识别这帧图片中的人体姿势数据，保存到self.results中
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img
    # 获取关键点坐标
    def findPosition(self,img,draw=True):
        '''
               获取人体姿势数据
               :param img: 一帧图像
               :param draw: 是否画出人体姿势节点和连接图
               :return: 人体姿势数据列表
               '''
        # 人体姿势数据列表，每个成员由3个数字组成：id, x, y
        # id代表人体的某个关节点，x和y代表坐标位置数据
        self.lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    # 人体姿势中3个点p1-p2-p3的角度
    def findAngle(self,img,p1,p2,p3,draw=True):
        '''
              获取人体姿势中3个点p1-p2-p3的角度
              :param img: 一帧图像
              :param p1: 第1个点
              :param p2: 第2个点
              :param p3: 第3个点
              :param draw: 是否画出3个点的连接图
              :return: 角度
              '''
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]

        """
        使用三角函数公式获取3个点p1-p2-p3，以p2为角的角度值，0-180度之间
        math.degrees(x) 方法将角度 x 从弧度转换为度数。
        PI (3.14..) 弧度等于 180 度，也就是说 1 弧度等于 57.2957795 度。
        """
        angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        #print(angle)
        if angle<0:
            #angle+=360
            angle=abs(angle)
        if angle>180:
            angle = abs(360-angle)

        #Draw
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2, y2),(255,255,255),3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img,str(int(angle)),(x2-50,y2+50),
                         cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return angle

def main():
    cap = cv2.VideoCapture('9.mp4')
    pTime = 0
    detector=poseDetector()
    while True:
        success, img = cap.read()
        img=detector.findPose(img)
        lmList=detector.findPosition(img)
        #print(lmList[14])
        #cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__=="__main__":
    main()
