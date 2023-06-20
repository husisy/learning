import time
import datetime
import random
import tenacity


# by default try forever without waiting when an exception is raised
@tenacity.retry
def hf0():
    if random.randint(0, 10) > 1:
        assert False
    else:
        return 233
x0 = hf0()


# function to be decorated
def hf1():
    print(datetime.datetime.now())
    time.sleep(random.uniform(0.3, 0.6))
    if random.randint(0, 10) > 1:
        assert False
    else:
        return 233

## retry max 3 times
hf1_r0 = tenacity.retry(stop=tenacity.stop_after_attempt(3))(hf1)
hf1_r0()
# RetryError if still fails after 3 times

## retry max 4 seconds
hf1_r0 = tenacity.retry(stop=tenacity.stop_after_delay(4))(hf1)
hf1_r0()


## if more than 3 times, or pass 4 seconds, stop retrying
hf1_r0 = tenacity.retry(stop=(tenacity.stop_after_delay(3) | tenacity.stop_after_attempt(4)))(hf1)
hf1_r0()


## wait 1 second between each retry
hf1_r0 = tenacity.retry(wait=tenacity.wait_fixed(1))(hf1)
hf1_r0()


hf1_r0 = tenacity.retry(wait=tenacity.wait_random(min=0.5, max=1))(hf1)
hf1_r0()
