import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.models import load_model
from socket import *

serverName = '192.168.0.5'
serverPort = 12000

PWMA = 18
AIN1   =  22
AIN2   =  27

PWMB = 23
BIN1   = 25
BIN2  =  24

def motor_back(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    
def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1

def motor_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,False) #BIN1
    
def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    
def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
        
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)

L_Motor= GPIO.PWM(PWMA,500)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB,500)
R_Motor.start(0)

speedSet = 30

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = list(exp_a / sum_exp_a)
    return y.index(max(y))
        
def main(): 

    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName,serverPort))

    model_path = '/home/pi/AI_CAR/lane_navigation_final(ResNet-50).h5'
    model = load_model(model_path) 

    carState = "stop"
    
    try:
        with open('socket_dalay.txt', 'w') as f:
            sys.stdout = f
            while True:
                keyValue = cv2.waitKey(1)
            
                if keyValue == ord('q') :
                    break
                elif keyValue == 82 :
                    print("go")
                    carState = "go"
                elif keyValue == 84 :
                    print("stop")
                    carState = "stop"

                
                start = time.time()
                _, image = camera.read()
                image = cv2.flip(image,-1)
                preprocessed = img_preprocess(image)
                # cv2.imshow('pre', preprocessed)

                # socket communication
                clientSocket.send(preprocessed.encode())
                clientSocket.close()
                
                X = np.asarray([preprocessed])
                direction = softmax(model.predict(X)[0])
                

                end = time.time()
                print(f"{end - start:.5f} sec")

                #print("direction:",direction)
                    
                if carState == "go":
                    if direction == 1:
                        print("go")
                        speedSet = 50
                        motor_go(speedSet)
                        # print(time.time() - start)
                    elif direction == 2:    
                        print("right")
                        speedSet = 40
                        motor_right(speedSet)
                        # print(time.time() - start)
                    elif direction == 0:
                        print("left")
                        speedSet = 40
                        motor_left(speedSet)
                        # print(time.time() - start)
                elif carState == "stop":
                    motor_stop()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
