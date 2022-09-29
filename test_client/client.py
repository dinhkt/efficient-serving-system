import requests, json, base64
import time
url = "http://localhost:8081/predict"



""" Client request format
    image: request image
    model: wish to use which model
    slo: expected backend inference time(ms)
"""
for _ in range(100):
    image_path = "dog.jpg"
    # image_path = "horses.jpg"
    # image_path = "both.png"
    st=time.time()
    result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"resnet18","slo":6}).text

    print(json.loads(result))
    print("r1:"+str(time.time()-st))
    st=time.time()
    # image_path = "dog.jpg"
    image_path = "horses.jpg"
    # image_path = "both.png"
    result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"resnet50","slo":8}).text

    print(json.loads(result))
    print("r2:"+str(time.time()-st))
    # st=time.time()
    # # image_path = "dog.jpg"
    # # image_path = "horses.jpg"
    # image_path = "both.png"
    # result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"vgg16","slo":10}).text

    # print(json.loads(result))
    # print("r3:"+str(time.time()-st))