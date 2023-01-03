import requests, json, base64
import time
from threading import Thread

url = "http://localhost:8082/process"


def send_request(model,slo,bs):
    print("send request")
    st=time.time()
    result = requests.post(url, json={"image0": base64.b64encode(open("image.jpeg", "rb").read()).decode('utf-8'),
                                        "image1": base64.b64encode(open("horses.jpg", "rb").read()).decode('utf-8'),
                                        "image2": base64.b64encode(open("both.png", "rb").read()).decode('utf-8'),
                                        "image3": base64.b64encode(open("dog.jpg", "rb").read()).decode('utf-8'),
                                        "image4": base64.b64encode(open("dog.jpg", "rb").read()).decode('utf-8'),
                                        "model":model,"slo":slo,"batchsize":bs,"service_type":0}).text
    print(json.loads(result))
    print("t:"+str(time.time()-st))

for _ in range(1):
    t1=Thread(target=send_request,args=("resnet18",30,4))
    t1.start()
    # t2=Thread(target=send_request,args=("resnet50",40,3))
    # t2.start()
    # t3=Thread(target=send_request,args=("vgg16",40,2))
    # t3.start()
# send_request("resnet18",30,4)


