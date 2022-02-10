import absl
import absl.app
import absl.flags

FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("name", None, "Your name.")
absl.flags.DEFINE_integer("num_times", 1, "Number of times to print greeting.")
absl.flags.DEFINE_string("my_integer0", None, "my_integer0")
absl.flags.mark_flag_as_required("name")

def main(argv):
    del argv  # Unused.
    print('[my_integer0]', FLAGS.my_integer0)
    for i in range(0, FLAGS.num_times):
        print('Hello, %s!' % FLAGS.name)

# python draft01.py --name=daenerys --num_times=2
if __name__ == '__main__':
    absl.app.run(main)
