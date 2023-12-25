import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms

# GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels= 24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.conv2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.conv3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = nn.functional.elu(self.conv1(x))
        x = nn.functional.elu(self.conv2(x))
        x = nn.functional.elu(self.conv3(x))
        x = nn.functional.elu(self.conv4(x))
        x = self.dropout1(x)
        x = nn.functional.elu(self.conv5(x))
        x = self.flatten(x)
        x = self.dropout2(x)
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.elu(self.fc3(x))
        x = self.output(x)
        return x

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

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

L_Motor= GPIO.PWM(PWMA,100)
L_Motor.start(0)

R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)

speedSet = 40

def img_preprocess(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((int(image.shape[0]/2), image.shape[1])),
        transforms.Resize((66, 200)),
        transforms.GaussianBlur((5,5),sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
    image = transform(image)
    
    return image

camera = cv2.VideoCapture(-1)
camera.set(3, 640)
camera.set(4, 480)

        
def main():
    
    model = NvidiaModel()
    model.load_state_dict(torch.load('/home/pi/AI_CAR/model/lane_navigation_final_Pytorch.pth', map_location=device))


    carState = "stop"
    
    #keyValue = input()
    
    try:
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
                
            
                
            _, image = camera.read()
            image = cv2.flip(image,-1)
            preprocessed = img_preprocess(image)
            
            imshow_image = preprocessed #imshow_image : for imshow, preserve 3D image
            imshow_image = imshow_image.detach().cpu().numpy() # tensor -> numpy
            imshow_image = np.transpose(imshow_image, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            imshow_image = imshow_image * 255.0
            imshow_image = cv2.cvtColor(imshow_image, cv2.COLOR_BGR2RGB) # RGB 
            cv2.imshow('w', imshow_image)

            
            preprocessed = preprocessed.unsqueeze(0) # 4D image(batch, C, H, W)
            
            # X = np.asarray([preprocessed])
            steering_angle = model(preprocessed) # inputs = X
            # steering_angle = int(model.predict(X)[0]) - tensorflow
            print("predict angle:",steering_angle)
                
            if carState == "go":
                if steering_angle >= 70 and steering_angle <= 110:
                    print("go")
                    motor_go(speedSet)
                elif steering_angle > 111:
                    print("right")
                    motor_right(speedSet)
                elif steering_angle < 71:
                    print("left")
                    motor_left(speedSet)
            elif carState == "stop":
                motor_stop()
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
