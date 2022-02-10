import time
import glob
import aiofile
import asyncio

all_file_path = glob.glob('/zcdata/val/*/*.JPEG')[:1000]
def hf0(all_file_path):
    ret = [open(x,'rb').read() for x in all_file_path]
    return ret

t0 = time.time()
z0 = hf0(all_file_path)
print(time.time()-t0)

async def hf1_read_one(filepath):
    async with aiofile.AIOFile(filepath, 'rb') as fid:
        return await fid.read()

async def hf1(all_file_path):
    z0 = [(hf1_read_one(x)) for x in all_file_path]
    z1 = [(await x) for x in z0]
    return z1

t0 = time.time()
z1 = asyncio.run(hf1(all_file_path)) #terrible time usage
print(time.time()-t0)

assert all(x==y for x,y in zip(z0,z1))
