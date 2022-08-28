import moderngl
from PyQt5 import QtWidgets
from pydelfem3.drawer_meshpos import DrawerMesPos, ElementInfo
import numpy

import pydelfem3.qtglwidget_viewer3
import pydelfem3


def draw_mesh(V,F):
    E = pydelfem3.edges_of_uniform_mesh(F, V.shape[0])

    with QtWidgets.QApplication([]) as app:
        drawer = DrawerMesPos(
            V=V.astype(numpy.float32),
            element=[
                ElementInfo(index=F.astype(numpy.uint32), color=(1, 0, 0), mode=moderngl.TRIANGLES),
                ElementInfo(index=E.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES)]
        )
        win = pydelfem3.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer])
        win.show()
        app.exec()


if __name__ == "__main__":
    draw_mesh( *pydelfem3.torus_meshtri3(0.6, 0.3, 32, 32) )
    draw_mesh( *pydelfem3.capsule_meshtri3(0.1, 0.6, 32, 32, 32) )
    draw_mesh( *pydelfem3.cylinder_closed_end_meshtri3(0.1, 0.8, 32, 32) )
    draw_mesh( *pydelfem3.sphere_meshtri3(1., 32, 32) )



