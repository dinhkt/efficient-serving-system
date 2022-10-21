import aiohttp
import asyncio
import time

url = "http://143.248.148.29:8082/predict"
rate=0.02
N=2000

async def upload1(session,idx):
    await asyncio.sleep(idx*rate)
    await session.post(url,json={"image": base64.b64encode(open("./dog.jpg", "rb").read()).decode('utf-8'),"model":"resnet18","slo":15})

async def main(**kwargs):
    # Asynchronous context manager.  Prefer this rather
    # than using a different session for each GET request
    async with aiohttp.ClientSession() as session:
        tasks = []
        for c in range(N):
            tasks.append(upload1(session=session,idx=c))
            tasks.append(upload2(session=session,idx=c))
            tasks.append(upload3(session=session,idx=c))
        # asyncio.gather() will wait on the entire task set to be
        # completed.  If you want to process results greedily as they come in,
        # loop over asyncio.as_completed()
        htmls = await asyncio.gather(*tasks, return_exceptions=True)
        return htmls

async def upload2(session,idx):
    await asyncio.sleep(5+idx*rate)
    await session.post(url,json={"image": base64.b64encode(open("./horses.jpg", "rb").read()).decode('utf-8'),"model":"resnet50","slo":22})


async def upload3(session,idx):
    await asyncio.sleep(10+idx*rate)
    await session.post(url,json={"image": base64.b64encode(open("./both.png", "rb").read()).decode('utf-8'),"model":"vgg16","slo":18})



if __name__ == '__main__':
    st=time.time()
    asyncio.run(main())  # Python 3.7+
    print("total",time.time()-st)