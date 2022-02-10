from pathlib import Path, PurePath

dir0 = Path('.')
list(dir0.iterdir())
list(dir0.glob('*.py'))
list(dir0.glob('*/*.py'))
[x for x in dir0.iterdir() if x.is_dir()]
[x for x in dir0.iterdir() if x.is_file()]

'tbd00'/dir0/'tbd01'

dir1 = dir0/'tbd00'
if not dir1.exists():
    dir1.mkdir()
file0 = dir1/'tbd00.txt'
file0.exists()
file0.is_absolute()
file0.is_dir()
file0.is_file()
file0.resolve()
file0.as_posix()
file0.resolve().as_uri()
file0.is_reserved() #la ji windows, see https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
file0.math('*.txt')
file0.with_name('tbd01.txt')
file0.with_suffix('.py')
# database_path.unlink() #delete file
dir0.joinpath('tbd00.txt')

file0.parts
file0.parent
file0.parents
file0.name
file0.suffix
file0.suffixes
file0.stem

file0.write_text('hello world\nha li lu ya\n233333\n33332\nshenmegui')
file0.read_text()
with file0.open() as fid:
    pass
# file0.read_bytes()
# file0.write_bytes()
# file0.touch()

str(file0)
Path(str(file0))
