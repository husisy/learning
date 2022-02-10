import turtle as t

if __name__ == "__main__":
    # seems that turtle can only run once
    t.pensize(8) #set line thickness

    t.forward(200)
    t.right(90)

    t.pencolor('red')
    t.forward(100)
    t.right(90)

    t.pencolor('green')
    t.forward(200)
    t.right(90)

    t.pencolor('blue')
    t.forward(100)

    t.done()
