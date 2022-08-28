import moderngl
import pyrr
from PyQt5 import QtWidgets, QtCore
from pydelfem3.drawer_meshpos import DrawerMesPos, ElementInfo
from pydelfem3.drawer_transform import DrawerTransformer
import numpy

from pathlib import Path

import pydelfem3.qtglwidget_viewer3
import pydelfem3


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        newpath = Path('.') / '..' / 'external' / 'delfem3' / 'glutin_examples' / 'asset' / 'HorseSwap.obj'
        vtx_xyz, elem_vtx_index, elem_vtx_xyz = pydelfem3.load_wavefront_obj(str(newpath))
        self.vtx_xyz = pydelfem3.centerize_scale_3d_points(vtx_xyz)
        elem_vtx_xyz = elem_vtx_xyz.astype(numpy.uint64)

        E = pydelfem3.edges_of_triquad_mesh(elem_vtx_index, elem_vtx_xyz, self.vtx_xyz.shape[0])
        self.tri_vtx = pydelfem3.triangles_from_triquad_mesh(elem_vtx_index, elem_vtx_xyz)

        drawer_triquadmesh3 = DrawerMesPos(
            V=self.vtx_xyz.astype(numpy.float32),
            element=[
                ElementInfo(index=E.astype(numpy.uint32), color=(0, 0, 0), mode=moderngl.LINES),
                ElementInfo(index=self.tri_vtx.astype(numpy.uint32), color=(1, 1, 1), mode=moderngl.TRIANGLES)]
        )

        V,F = pydelfem3.sphere_meshtri3(1., 32, 32)
        self.drawer_sphere = DrawerMesPos(V, element=[
            ElementInfo(index=F.astype(numpy.uint32), color=(1.,0.,0.), mode=moderngl.TRIANGLES)])
        self.drawer_sphere = DrawerTransformer(self.drawer_sphere)
        self.drawer_sphere.transform = pyrr.Matrix44.from_scale((0.05,0.05,0.05))

        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.glwidget = pydelfem3.qtglwidget_viewer3.QtGLWidget_Viewer3([drawer_triquadmesh3, self.drawer_sphere])
        self.glwidget.mousePressCallBack.append(self.mouse_press_callback)
        self.setCentralWidget(self.glwidget)

    def mouse_press_callback(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
            return
        src, dir = self.glwidget.nav.picking_ray()
        pos, tri_index = pydelfem3.first_intersection_ray_meshtri3(
            numpy.array(src.xyz).astype(numpy.float32), numpy.array(dir.xyz).astype(numpy.float32),
            self.vtx_xyz, self.tri_vtx)
        self.drawer_sphere.is_visible = False
        if tri_index != -1:
            self.drawer_sphere.is_visible = True
            self.drawer_sphere.transform = pyrr.Matrix44.from_translation(pos) * pyrr.Matrix44.from_scale((0.03, 0.03, 0.03))
        self.glwidget.updateGL()




if __name__ == "__main__":

    with QtWidgets.QApplication([]) as app:
        win = MainWindow()
        win.show()
        app.exec()




