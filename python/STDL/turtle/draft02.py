import turtle as t


def get_pencolor(_PEN_COLOR=[0, 0, 0]):
    ret = tuple(_PEN_COLOR)
    _PEN_COLOR[0] = (_PEN_COLOR[0] + 2) % 200
    _PEN_COLOR[1] = (_PEN_COLOR[1] + 4) % 200
    _PEN_COLOR[2] = (_PEN_COLOR[2] + 6) % 200
    return ret


def draw_tree(len_, level):
    w = t.pensize()

    t.pensize(w * 0.75)
    t.pencolor(get_pencolor())

    t.left(45)
    t.forward(len_)
    if level > 0:
        draw_tree(len_*0.75, level-1)
    t.backward(len_)

    t.right(90)
    t.forward(len_)
    if level > 0:
        draw_tree(len_*0.75, level - 1)
    t.backward(len_)

    t.left(45)
    t.pensize(w)


if __name__ == "__main__":
    t.speed('fastest')
    # set color mode to integer [R,G,B], default is float [R,G,B]
    t.colormode(255)

    t.left(90)  # heading up
    t.pensize(10)
    t.pencolor(get_pencolor())
    len_ = 120
    t.penup()
    t.backward(len_)
    t.pendown()
    t.forward(len_)
    draw_tree(len_*0.7, 9)

    t.done()
