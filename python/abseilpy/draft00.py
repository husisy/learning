import sys
import absl
import absl.flags
import absl.app

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string('echo', None, 'Text to echo.')


def main(argv):
    del argv  # Unused.

    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info), file=sys.stderr)
    absl.logging.info('echo is %s.', FLAGS.echo)


# python draft00.py --echo smoke
if __name__ == '__main__':
    absl.app.run(main)
