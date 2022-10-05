import requests, json, base64
import time
from threading import Thread

url = "http://localhost:8082/predict"


def send_request(img_path,model,slo,bs):
    print("send request")
    st=time.time()
    result = requests.post(url, json={"image": base64.b64encode(open(img_path, "rb").read()).decode('utf-8'),"model":model,"slo":slo}).text
    print(json.loads(result))
    print("t:"+str(time.time()-st))

""" Client request format
    image: request image
    model: wish to use which model
    slo: expected backend inference time(ms)
"""

t1=Thread(target=send_request,args=("dog.jpg","resnet18",10,4))
t1.start()
t2=Thread(target=send_request,args=("horses.jpg","resnet50",15,2))
t2.start()
t3=Thread(target=send_request,args=("image.jpeg","vgg16",10,1))
t3.start()


