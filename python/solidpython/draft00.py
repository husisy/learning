import numpy as np

import solid2

def demo00():
    model = solid2.cube(4)
    model.save_as_scad('tbd00.scad')


def demo01():
    solid2.set_global_fn(100) #the number of faces for curved shapes, default (TODO)

    # create enclosure base
    base = solid2.cube(100, 50, 20)
    hole = solid2.cube(90, 40, 20).translate(5, 5, 5)
    base = base - hole

    # create enclosure lid
    lid = solid2.cube(100, 50, 5).translate(0, 0, 0)
    label = solid2.text('box').linear_extrude(height=1).translate(5, 5, 5)
    lid = lid + label

    # create reusable screw hole function
    def screw_hole():
        head = solid2.cylinder(2, 2.5)
        body = solid2.cylinder(1, 10)
        return (head + body).mirror(0, 0, 1)

    # cut out screw holes
    offset = 3
    lid -= screw_hole().translate(3, 3, 0.5)
    lid -= screw_hole().translate(3, 47, 0.5)
    lid -= screw_hole().translate(97, 3, 0.5)
    lid -= screw_hole().translate(97, 47, 0.5)

    base -= screw_hole().translate(3, 3, 23)
    base -= screw_hole().translate(3, 47, 23)
    base -= screw_hole().translate(97, 3, 23)
    base -= screw_hole().translate(97, 47, 23)

    # move lid into position, make lid transparent
    lid = lid.translate(0, 0, 20).background()
    model = base + lid
    model.save_as_scad('tbd00.scad')
