import manim

class CreateCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()
        circle.set_fill(manim.PINK, opacity=0.5) #transparency=1-opacity
        self.play(manim.Create(circle))
# manim -pql draft00.py CreateCircle

class SquareToCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()
        circle.set_fill(manim.PINK, opacity=0.5)

        square = manim.Square()
        square.rotate(manim.PI/4)

        self.play(manim.Create(square))
        self.play(manim.Transform(square, circle))
        self.play(manim.FadeOut(square))
# manim -pql draft00.py SquareToCircle

class SquareAndCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()
        circle.set_fill(manim.PINK, opacity=0.5)

        square = manim.Square()
        square.set_fill(manim.BLUE, opacity=0.5)

        square.next_to(circle, manim.RIGHT, buff=0.5) #manim.LEFT manim.UP manim.DOWN
        self.play(manim.Create(circle), manim.Create(square))
# manim -pql draft00.py SquareAndCircle

class AnitatedSquareToCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()
        square = manim.Square()

        self.play(manim.Create(square))
        self.play(square.animate.rotate(manim.PI/4))
        self.play(manim.ReplacementTransform(square, circle))
        self.play(circle.animate.set_fill(manim.PINK, opacity=0.5))
# manim -pql draft00.py AnitatedSquareToCircle


class DifferentRotations(manim.Scene):
    def construct(self):
        left_square = manim.Square(color=manim.BLUE, fill_opacity=0.7).shift(2*manim.LEFT)
        right_square = manim.Square(color=manim.GREEN, fill_opacity=0.7).shift(2*manim.RIGHT)
        self.play(left_square.animate.rotate(manim.PI), manim.Rotate(right_square, angle=manim.PI), run_time=2)
        self.wait()
# manim -pql draft00.py DifferentRotations
