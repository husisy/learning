import argparse


def demo_basic():
    parser = argparse.ArgumentParser(description='demo-basic argparser')
    parser.add_argument('x', type=int, help='required argument x')
    parser.add_argument('-y', '--yvalue', type=int, default=233, help='optional argument y')
    # parser.add_argument('move', choices=['rock', 'paper', 'scissors'])

    try:
        # default to sys.argv #sys.argv include xxx.py
        # equivalent ['-h'], ['--help']
        parser.parse_args(['--help']) #print the help information then exit
    except SystemExit:
        pass

    args = parser.parse_args(['23'])
    assert args.x==23
    assert args.yvalue==233 #default
    args = parser.parse_args(['23', '-y', '-233'])
    assert args.x==23
    assert args.yvalue==-233
    args = parser.parse_args(['23', '--yvalue', '-2333'])
    assert args.x==23
    assert args.yvalue==-2333


def demo_array():
    parser = argparse.ArgumentParser(description='demo-basic01 argparser')
    parser.add_argument('x1', metavar='N', type=int, nargs='+', help='integer array')
    parser.add_argument('--sum', dest='hf1', action='store_const',
            const=sum, default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args(['--sum', '7', '-1', '42'])
    assert args.hf1(args.x1)==48


def demo_true_false():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true') #True if supplied, otherwise False
    parser.add_argument('-b', action='store_true')
    parser.add_argument('-c', action='store_false') #False if supplied, otherwise True
    parser.add_argument('-d', action='store_false')
    # parser.add_argument('-e', action='store_true', default=True) #useless, always True
    args = parser.parse_args(['-ac'])
    assert args.a
    assert not args.b
    assert not args.c
    assert args.d
