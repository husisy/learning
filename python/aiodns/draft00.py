import asyncio
import aiodns

loop = asyncio.get_event_loop()
resolver = aiodns.DNSResolver(loop=loop)

async def query(name, query_type):
    return await resolver.query(name, query_type)

coro = query('bing.com', 'A')
result = loop.run_until_complete(coro)
print(result)
