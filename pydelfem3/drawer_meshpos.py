import typing
from pyrr import Matrix44
import numpy
import moderngl

class ElementInfo:

    def __init__(self, index, mode, color):
        self.vao = None
        self.index = index
        self.mode = mode
        self.color = color

class DrawerMesPos:

    def __init__(self, V:numpy.ndarray, element: typing.List[ElementInfo]):
        self.V = V
        self.element = element
        self.vao_content = None

    def init_gl(self, ctx: moderngl.Context):
        self.prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 color;                
                out vec4 f_color;
                void main() {
                    f_color = vec4(color, 1.0);
                }
            '''
        )
        self.uniform_mvp = self.prog['Mvp']
        self.uniform_color = self.prog['color']

        self.vao_content = [
            (ctx.buffer(self.V.tobytes()), '3f', 'in_position'),
        ]
        del self.V
        for el in self.element:
            index_buffer = ctx.buffer(el.index.tobytes())
            el.vao = ctx.vertex_array(
                self.prog, self.vao_content, index_buffer, 4
            )
            del el.index

    def update_position(self, V: numpy.ndarray):
        if self.vao_content != None:
            vbo = self.vao_content[0][0]
            vbo.write(V.tobytes())

    def paint_gl(self, mvp: Matrix44):
        self.uniform_mvp.value = tuple(mvp.flatten())
        for el in self.element:
            self.uniform_color.value = el.color
            el.vao.render(mode=el.mode)