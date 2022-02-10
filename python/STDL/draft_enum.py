import enum


@enum.unique #otherwise: Sun=0 Mon=0 # treat as alias
class Weekday(enum.Enum):
    Sun = 0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6

repr(Weekday.Sun)
str(Weekday.Sun)
len(Weekday)

Weekday.Sun == Weekday.Mon #False
Weekday.Sun == Weekday.Sun #True
# !=
Weekday.Sun is Weekday.Mon #False
Weekday.Sun is Weekday.Sun #True
# is not

Weekday.Sun.name
Weekday.Sun.value
Weekday(Weekday.Sun.value)
Weekday[Weekday.Sun.name]

list(Weekday)
_ = {x:x.value for x in Weekday} #self:value
_ = {x.name:x.value for x in Weekday} #name:value

@enum.unique
class Color(enum.Enum):
    RED = enum.auto()
    GREEN = enum.auto()
    BLUE = enum.auto()
