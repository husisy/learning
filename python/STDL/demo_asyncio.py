import time
import asyncio
from datetime import datetime


async def delayed_say(delay, what):
    print(f"delayed_say({delay}, {what}) started at {time.strftime('%X')}")
    await asyncio.sleep(delay)
    print(f"delayed_say({delay}, {what}) finished at {time.strftime('%X')}")
    return 233


def demo_asyncio_run():
    tmp0 = delayed_say(1, 'demo_asyncio_run')
    tmp1 = asyncio.run(tmp0)
    print('demo_asyncio_run:: return_value:', tmp1)


async def demo_await():
    print(f"demo_await() started at {time.strftime('%X')}")
    tmp0 = delayed_say(2, 'hello')
    tmp1 = delayed_say(1, 'world')
    await tmp0
    print('what')
    await tmp1
    print(f"demo_await() finished at {time.strftime('%X')}")


async def demo_create_task():
    print(f"demo_create_task() started at {time.strftime('%X')}")
    task1 = asyncio.create_task(delayed_say(2, 'hello'))
    task2 = asyncio.create_task(delayed_say(1, 'world'))
    await task1 #when calling await, all awaitables will be running
    print('what')
    await task2
    print(f"demo_create_task() finished at {time.strftime('%X')}")


async def cancelable_task():
    print(f"cancelable_task() started at {time.strftime('%X')}")
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print(f"cancelable_task() canceled at {time.strftime('%X')}")
        raise
    finally:
        print(f"cancelable_task() finished at {time.strftime('%X')}")


async def demo_cancel_task():
    task = asyncio.create_task(cancelable_task())
    await asyncio.sleep(1) #hf1_1 is running
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print(f"demo_cancel_task:: cancelable_task() canceled at {time.strftime('%X')}")


if __name__ == "__main__":
    demo_asyncio_run()
    print()
    asyncio.run(demo_await())
    print()
    asyncio.run(demo_create_task())
    print()
    asyncio.run(demo_cancel_task())
