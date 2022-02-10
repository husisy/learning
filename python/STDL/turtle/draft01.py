import turtle as t

def draw_star(x, y):
    t.penup()
    t.goto(x, y)
    t.pendown()

    t.setheading(0)
    for _ in range(5):
        t.forward(40)
        t.right(144)


if __name__ == "__main__":
    for x in range(0, 250, 50):
        draw_star(x, 0)
    t.done()
