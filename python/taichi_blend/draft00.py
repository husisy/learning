import numpy as np
import bpy
import numblend as nb

nb.init()

N = 64

for x in bpy.data.objects:
    nb.delete_object(x.name)

objects = []
for x in range(N):
    bpy.ops.mesh.primitive_cube_add(size=1)
    objects.append(bpy.context.object)

tmp0 = np.arange(N)
tmp1 = np.zeros(N)
locations = np.stack([tmp0-N/2, tmp1, np.sin(tmp0*0.3)], axis=1)
#nb.objects_update(objects[i], location=locations[i])
callback = nb.objects_update(objects, location=locations)
callback()

@nb.add_animation
def main():
    for ind_frame in range(250):
        locations = np.stack([tmp0-N/2, tmp1, np.sin((tmp0+ind_frame)*0.3)], axis=1)
        yield nb.objects_update(objects, location=locations)
