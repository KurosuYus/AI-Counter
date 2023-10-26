import videoprocess as vp
import trainingsetprocess as tp
import videocapture as vc
import cv2
import mhand


camera = cv2.VideoCapture(0)




if __name__ == '__main__':
    while True:

        print("请输入检测模式（数字）：1. 从本地导入视频检测\t2. 调用摄像头检测\t3. 退出\n")
        menu = mhand.beginOK(camera, [1,2,3])
        
        #print(menu)
        #menu = int(input("请输入检测模式（数字）：1. 从本地导入视频检测\t2. 调用摄像头检测\t3. 退出\n"))
        if menu == 1:
            #flag = int(input("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n"))
            print("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n")
            
            flag = mhand.beginOK(camera, [1,2,3,4])
            #video_path = input("请输入视频路径：")
            tp.trainset_process(flag)
            vp.video_process(video_path, flag)
            continue
        elif menu == 2:
            # flag = int(input("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n"))
            # print("\n按键q或esc退出摄像头采集")
           
            print("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n")
            flag = mhand.beginOK(camera, [2])
            #mhand.beginOK(camera)
            tp.trainset_process(flag)
            vc.process(flag)
            continue
        elif menu == 3:
            print("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n")
            flag = mhand.beginOK(camera, [3])
            # mhand.beginOK(camera)
            tp.trainset_process(flag)
            vc.process(flag)
            continue
        elif menu == 4:
            print("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上\t4. 哑铃\n")
            flag = mhand.beginOK(camera, [4])
            # mhand.beginOK(camera)
            tp.trainset_process(flag)
            vc.process(flag)
            continue
        else:
            print("输入错误，请重新输入！")
            continue




