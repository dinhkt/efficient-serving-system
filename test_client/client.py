import requests, json, base64

url = "http://localhost:8081/predict"

# image_path = "image.jpeg"
image_path = "dog.jpg"
# image_path = "horses.jpg"
# image_path = "both.png"

""" Client request format
    image: request image
    model: wish to use which model
    slo: expected backend inference time(ms)
"""
result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"resnet18","slo":3}).text

print(json.loads(result))

# image_path = "dog.jpg"
image_path = "horses.jpg"
# image_path = "both.png"
result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"resnet50","slo":8}).text

print(json.loads(result))

# image_path = "dog.jpg"
# image_path = "horses.jpg"
image_path = "both.png"
result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode('utf-8'),"model":"vgg16","slo":10}).text

print(json.loads(result))