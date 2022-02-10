import click


@click.command()
@click.option('--count', default=1, help='number of greetings.')
@click.option('--name', prompt='your name', help='the person to greet')
def hello(count, name):
    for _ in range(count):
        print('hello {}!'.format(name))

# try:
#     python draft00.py --help
#     python draft00.py --count 2
#     python draft00.py --name world
if __name__=='__main__':
    hello()
