# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false

import seekfree, pyb
import sensor, image, time, tf, gc

sensor.reset()                      # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
clock = time.clock()                # Create a clock object to track the FPS.

#设置模型路径
face_detect = '/sd/yolo.tflite'
#载入模型
net = tf.load(face_detect)

while(True):
    clock.tick()
    img = sensor.snapshot()
    img.lens_corr(strength=2.8, zoom=1.0)

    img1 = img.copy(0.75, 1)
    #使用模型进行识别
    for obj in tf.detect(net,img1):
        x1,y1,x2,y2,label,scores = obj

        if(scores>0.90):
            print(obj)
            w = x2- x1
            h = y2 - y1
            x1 = int((x1)*img.width())
            y1 = int(y1*img.height())
            w = int(w*img.width())
            h = int(h*img.height())

            if(label == 0):
                img.draw_string(x1, y1-15, "tennis", color = (191, 255, 0), scale = 2, mono_space = False)
                img.draw_rectangle((x1,y1,w,h),color=(191, 255, 0),thickness=2)
            elif(label == 1):
                img.draw_string(x1, y1-15, "red", color = (255, 0, 0), scale = 2, mono_space = False)
                img.draw_rectangle((x1,y1,w,h),color=(255, 0, 0),thickness=2)
            elif(label == 2):
                img.draw_string(x1, y1-15, "blue", color = (0, 0, 255), scale = 2, mono_space = False)
                img.draw_rectangle((x1,y1,w,h),color=(0, 0, 255),thickness=2)
            elif(label == 3):
                img.draw_string(x1, y1-15, "brown", color = (80, 40, 20), scale = 2, mono_space = False)
                img.draw_rectangle((x1,y1,w,h),color=(80, 40, 20),thickness=2)
            elif(label == 4):
                img.draw_string(x1, y1-15, "white", color = (255, 255, 255), scale = 2, mono_space = False)
                img.draw_rectangle((x1,y1,w,h),color=(255, 255, 255),thickness=2)
    print(clock.fps())
    img.flush()
