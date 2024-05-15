-- this is a comment

-- lean --run draft00.lean

-- def main : IO Unit := IO.println "Hello, world!"

#eval (1+2*3 :Nat)
#eval (1-2 :Int)
#check (1-2 :Nat)

#eval String.append "Hello," " Lean!"
#eval String.append "Hello" (String.append ", " "Lean!")
#eval String.append "it is " (if 1>2 then "true" else "false")

def x0 := "Hello, "
def x1 :String := "Lean"
#eval String.append x0 x1

def hf0 (x :Int) :Int := x+1
#eval hf0 3
def hf1 (x :Int) (y :Int) :Int := if x>y then x else y
#eval hf1 3 4

def Str :Type := String
def x2 :Str := "Hello, "

-- foldable, unfoldable, reducible
def NaturalNumber :Type := Nat
abbrev NN :Type := Nat

structure Point where
  x :Float
  y :Float
deriving Repr

def x3 :Point := {x := 2.3, y := 2.33}
#eval x3
#eval x3.x
#eval ({x := 2.3, y := 2.33} :Point)
#eval {x := 2.3, y := 2.33 :Point}
def addPoints (p1 :Point) (p2 :Point) :Point := {x := p1.x + p2.x, y := p1.y + p2.y}
#eval addPoints {x := 1.0, y := 2.0} {x := 3.0, y := 4.0}
def distance (p1 :Point) (p2 :Point) :Float := Float.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
#eval distance {x := 1.0, y := 2.0} {x := 3.0, y := 4.0}

def zeroX (p :Point) :Point := {p with x := 0.0}
def x4 :Point := zeroX x3
#eval x3
#eval x4

#eval Bool.true

inductive MyBool where
  | myTrue :MyBool
  | myFalse :MyBool

inductive MyNat where
  | myZero :MyNat
  | mySucc (n :MyNat) :MyNat
deriving Repr

#eval MyNat.mySucc (MyNat.mySucc MyNat.myZero)

def isZero (n :Nat) :Bool :=
  match n with
   | Nat.zero => Bool.true
   | Nat.succ _ => Bool.false

#eval isZero (0 :Nat)
#eval isZero (1 :Nat)

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout

  stdout.putStrLn "How would you like to be addressed?"
  let input ← stdin.getLine
  let name := input.dropRightWhile Char.isWhitespace

  stdout.putStrLn s!"Hello, {name}!"
