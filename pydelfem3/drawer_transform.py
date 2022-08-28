from pyrr import Matrix44
import numpy
import moderngl

class DrawerTransformer:

    def __init__(self, drawer):
        self.transform = Matrix44.identity(dtype=numpy.float32)
        self.is_visible = True
        self.drawer = drawer

    def init_gl(self, ctx: moderngl.Context):
        self.drawer.init_gl(ctx)

    def paint_gl(self, mvp: Matrix44):
        if not self.is_visible:
            return
        self.drawer.paint_gl(mvp * self.transform)