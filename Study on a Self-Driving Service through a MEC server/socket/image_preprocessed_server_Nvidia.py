from socket import *
import numpy
import cv2
from tensorflow.keras.models import load_model
import time

# socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image,(5,5),0)
    _,image = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

# Client와 socket 연결하는 작업
serverPort = 12001
serverSocket = socket(AF_INET,SOCK_STREAM)
serverSocket.bind(('',serverPort))
serverSocket.listen(1)
print('The server is ready to receive')

# Socket connection
connectionSocket, addr = serverSocket.accept()
# 파일 경로 바꾸기 
model_path = 'lane_navigation_final(Nvidia).h5'
model = load_model(model_path)

start = time.time()
length = recvall(connectionSocket, 16) 
stringData = recvall(connectionSocket, int(length))
data = numpy.fromstring(stringData, dtype='uint8')

frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
print(frame)

A = time.time()
preprocessed = img_preprocess(frame)
B = time.time()

X = numpy.asarray([preprocessed])
steering_angle = int(model.predict(X)[0])
C = time.time()
# print("predict angle:", steering_angle)

print("local socket : ", A - start)
print("preprocessed : ", B - A)
print("model : ", C - B)
# connectionSocket.send(str(steering_angle).encode())
connectionSocket.close()