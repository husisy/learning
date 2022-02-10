import taichi as ti

# ti.cpu default
# ti.gpu fail on windows @20200619
# ti.opengl fail on ubuntu @20200619
# ti.metal apple metal backend
# ti.init(device_memory_GB=3.4)
# ti.init(default_fp=ti.f64) #[ti.i32] ti.i64 [ti.f32] ti.f64
ti.init(arch=ti.cpu)

n = 320
pixels = ti.field(dtype=ti.f32, shape=(n*2,n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
