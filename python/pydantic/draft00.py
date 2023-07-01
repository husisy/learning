import datetime
import pydantic

class DummyClass00(pydantic.BaseModel):
    a: int
    b: list[str]

x0 = DummyClass00(a=123, b=['abc', 'def'])
x0.model_dump() #dict
# {'a': 123, 'b': ['abc', 'def']}


class DummyClass01(pydantic.BaseModel):
    id: int
    name: str = '233'
    signup_ts: datetime.datetime | None
    tastes: dict[str, pydantic.PositiveInt]

tmp0 = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',
    'tastes': {
        'wine': 9,
        b'cheese': 7,
        'cabbage': '1',
    },
}
# import dateutil
# import dateutil.parser
# dateutil.parser.parse('2019-06-01 12:22')
x0 = DummyClass01(**tmp0)
x0.id
x0.signup_ts
x0.model_dump()
# {'id': 123,
#  'name': '233',
#  'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
#  'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1}}

## Error if validation fails
# tmp0 = {'id': 'not an int', 'tastes': {}}
# DummyClass01(**tmp0)
