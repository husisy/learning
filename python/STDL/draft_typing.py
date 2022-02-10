from typing import List, Dict, Tuple, Sequence

def hf0(x0:str) -> str:
    return 'hello' + x0
_ = hf0('233')


Vector = List[float]
def hf1(x0:float, x1:Vector) -> Vector:
    return [x0*y for y in x1]
_ = hf1(3, [1.0,2.0])


def hf2(x0: Tuple[str, int, Dict[str,str]]) -> Tuple[Dict[str,str], int, str]:
    return x0[2], x0[1], x0[0].upper()
_ = hf2(('abc', 233, {'abc':'233'}))
