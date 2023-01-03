import aiohttp
import asyncio
import time

url = "http://localhost:8082/process"
rate=0.02
N=200

async def upload(session,idx,model,slo,bs):
    await asyncio.sleep(idx*rate)
    await session.post(url,json={"image0": base64.b64encode(open("image.jpeg", "rb").read()).decode('utf-8'),
                                        "image1": base64.b64encode(open("horses.jpg", "rb").read()).decode('utf-8'),
                                        "image2": base64.b64encode(open("both.png", "rb").read()).decode('utf-8'),
                                        "image3": base64.b64encode(open("dog.jpg", "rb").read()).decode('utf-8'),
                                        "image4": base64.b64encode(open("dog.jpg", "rb").read()).decode('utf-8'),
                                        "model":model,"slo":slo,"batchsize":bs,"service_type":0})


async def main(**kwargs):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for c in range(N):
            tasks.append(upload(session=session,idx=c,model="resnet18",slo=30,bs=4))
            tasks.append(upload(session=session,idx=c,model="resnet50",slo=40,bs=3))
            tasks.append(upload(session=session,idx=c,model="vgg16",slo=40,bs=2))
        htmls = await asyncio.gather(*tasks, return_exceptions=True)
        return htmls

if __name__ == '__main__':
    st=time.time()
    asyncio.run(main())  # Python 3.7+
    print("total",time.time()-st)