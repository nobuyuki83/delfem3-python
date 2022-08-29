// use numpy::ndarray::{Array2, ArrayD, ArrayViewD, ArrayViewMutD};
// use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArray2, PyArray1};
use numpy::{IntoPyArray,
            PyReadonlyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
            PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use delfem3;

/// A Python module implemented in Rust.
#[pymodule]
fn pydelfem3(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    fn edges_of_uniform_mesh<'a>(
        py: Python<'a>,
        elems: PyReadonlyArray2<'a, usize>,
        num_vtx: usize) -> &'a PyArray2<usize> {
        let mshline = delfem3::msh_topology_uniform::mshline(
            &elems.as_slice().unwrap(), 3,
            &[0,1,1,2,2,0], num_vtx);
        numpy::ndarray::Array2::from_shape_vec(
            (mshline.len()/2,2), mshline).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    fn edges_of_triquad_mesh<'a>(
        py: Python<'a>,
        elem_ind: PyReadonlyArray1<'a, usize>,
        elem_vtx: PyReadonlyArray1<'a, usize>,
        num_vtx: usize) -> &'a PyArray2<usize> {
        let mshline = delfem3::msh_topology_mix::meshline_from_meshtriquad(
            &elem_ind.as_slice().unwrap(), &elem_vtx.as_slice().unwrap(), num_vtx);
        numpy::ndarray::Array2::from_shape_vec(
            (mshline.len()/2,2), mshline).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    fn triangles_from_triquad_mesh<'a>(
        py: Python<'a>,
        elem_ind: PyReadonlyArrayDyn<'a, usize>,
        elem_vtx: PyReadonlyArrayDyn<'a, usize>)  -> &'a PyArray2<usize> {
        let tri_vtx = delfem3::msh_topology_mix::meshtri_from_meshtriquad(
            &elem_ind.as_slice().unwrap(), &elem_vtx.as_slice().unwrap());
        numpy::ndarray::Array2::from_shape_vec(
            (tri_vtx.len()/3,3), tri_vtx).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    fn torus_meshtri3(
        py: Python,
        radius: f64, radius_tube: f64,
        nlg: usize, nlt: usize) -> (&PyArray2<f64>, &PyArray2<usize>) {
        let (vtx_xyz, tri_vtx) = delfem3::msh_primitive::torus_tri3::<f64>(
            radius, radius_tube, nlg, nlt);
        let v = numpy::ndarray::Array2::from_shape_vec(
            (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
        let f = numpy::ndarray::Array2::from_shape_vec(
            (tri_vtx.len()/3,3), tri_vtx).unwrap();
        (v.into_pyarray(py), f.into_pyarray(py))
    }

    #[pyfn(m)]
    fn capsule_meshtri3(
        py: Python,
        r: f64, l: f64,
        nc: usize, nr: usize, nl: usize) -> (&PyArray2<f64>, &PyArray2<usize>) {
        let (vtx_xyz, tri_vtx) = delfem3::msh_primitive::capsule_tri3::<f64>(
            r, l, nc, nr, nl);
        let v = numpy::ndarray::Array2::from_shape_vec(
            (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
        let f = numpy::ndarray::Array2::from_shape_vec(
            (tri_vtx.len()/3,3), tri_vtx).unwrap();
        (v.into_pyarray(py), f.into_pyarray(py))
    }

    #[pyfn(m)]
    fn cylinder_closed_end_meshtri3(
        py: Python,
        r: f64, l: f64,
        nr: usize, nl: usize) -> (&PyArray2<f64>, &PyArray2<usize>) {
        let (vtx_xyz, tri_vtx) = delfem3::msh_primitive::cylinder_closed_end_tri3::<f64>(
            r, l, nr, nl);
        let v = numpy::ndarray::Array2::from_shape_vec(
            (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
        let f = numpy::ndarray::Array2::from_shape_vec(
            (tri_vtx.len()/3,3), tri_vtx).unwrap();
        (v.into_pyarray(py), f.into_pyarray(py))
    }

    #[pyfn(m)]
    fn sphere_meshtri3(
        py: Python,
        r: f32,
        nr: usize, nl: usize) -> (&PyArray2<f32>, &PyArray2<usize>) {
        let (vtx_xyz, tri_vtx) = delfem3::msh_primitive::sphere_tri3(
            r, nr, nl);
        let v = numpy::ndarray::Array2::from_shape_vec(
            (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
        let f = numpy::ndarray::Array2::from_shape_vec(
            (tri_vtx.len()/3,3), tri_vtx).unwrap();
        (v.into_pyarray(py), f.into_pyarray(py))
    }

    #[pyfn(m)]
    fn load_wavefront_obj(
        py: Python,
        fpath: String) -> (&PyArray2<f32>, &PyArray1<usize>, &PyArray1<i32>) {
        let mut obj = delfem3::msh_io_obj::WavefrontObj::<f32>::new();
        obj.load(fpath.as_str());
        (
            numpy::ndarray::Array2::from_shape_vec(
                (obj.vtx_xyz.len()/3,3), obj.vtx_xyz).unwrap().into_pyarray(py),
            numpy::ndarray::Array1::from_vec(obj.elem_vtx_index).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(obj.elem_vtx_xyz).into_pyarray(py)
        )
    }

    #[pyfn(m)]
    fn first_intersection_ray_meshtri3<'a>(
        py: Python<'a>,
        src: PyReadonlyArray1<'a, f32>,
        dir: PyReadonlyArray1<'a, f32>,
        vtx_xyz: PyReadonlyArray2<'a, f32>,
        tri_vtx: PyReadonlyArray2<'a, usize>) -> (&'a PyArray1<f32>, i64)
    {
        use crate::delfem3::srch_bruteforce;
        let res = srch_bruteforce::intersection_meshtri3(
            src.as_slice().unwrap(),
            dir.as_slice().unwrap(),
            vtx_xyz.as_slice().unwrap(),
            tri_vtx.as_slice().unwrap());
        match res {
            None => {
                let a = PyArray1::<f32>::zeros(py,3,true);
                return (a, -1);
            },
            Some(postri) => {
                let a = PyArray1::<f32>::from_slice(py, &postri.0);
                return (a, postri.1 as i64);
            }
        }
    }

    Ok(())
}