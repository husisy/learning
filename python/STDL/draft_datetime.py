import time
import datetime

def utc2local(utc):
    # https://stackoverflow.com/a/19238551
    epoch = time.mktime(utc.timetuple())
    offset = datetime.datetime.fromtimestamp(epoch) - datetime.datetime.utcfromtimestamp(epoch)
    return utc + offset


x1 = datetime.datetime.now()
str(x1)

x1 = datetime.datetime(2018, 12, 16, 12, 0)
x2 = datetime.datetime(2018, 12, 16, 12, 1)
x2.timestamp() - x1.timestamp()

datetime.datetime.fromtimestamp(0)
datetime.datetime.utcfromtimestamp(0)

# see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
x1 = datetime.datetime.strptime('2018-12-16 12:01:02', '%Y-%m-%d %H:%M:%S')
x1.strftime('%a, %b, %d %H:%M')

x1 = datetime.datetime(2018, 12, 16, 12, 0)
x1 + datetime.timedelta(hours=10)
x1 - datetime.timedelta(days=1)

x1 = datetime.datetime.fromtimestamp(0)
x2 = x1.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=3)))

x1 = datetime.datetime.utcnow()
x2 = x1.replace(tzinfo=datetime.timezone.utc)
x3 = x2.astimezone(datetime.timezone(datetime.timedelta(hours=8)))

z0 = datetime.datetime.now()
z0.strftime('%Y%m%d-%H:%M:%S')
