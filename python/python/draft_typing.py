# python3.10+
from typing import Annotated

## simple types
int
float
bool
bytes

## generic types
list[str] #all elements are str
tuple[int, int, str] #tuple of 3 elements
set[bytes] #all elements are bytes
dict[str, float] #all keys are str, all values are float
int | str #either int or str
bytes | None #either bytes or None

## class as types

## metadata annotations
Annotated[int, 'positive']
