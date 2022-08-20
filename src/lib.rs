use numpy::ndarray::{Array2, ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use delfem3;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "pydelfem3")]
fn delfem3_python(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    fn sum_as_string(_py: Python, a:i64, b:i64) -> PyResult<String> {
        Ok(format!("{}", a + b))
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let mut x = unsafe { x.as_array_mut() };
        x *= a;
    }

    #[pyfn(m)]
    fn edges<'a>(
        py: Python<'a>,
        elems: PyReadonlyArrayDyn<'a, usize>,
        a: usize) -> &'a PyArray2<usize> {
        let mshline;
        {
            let elsup = delfem3::msh_topology_uniform::elsup(
                &elems.as_slice().unwrap(), elems.len()/3, 3, a);
            let psup = delfem3::msh_topology_uniform::psup_elem_edge(
                &elems.as_slice().unwrap(), 3,
                3, &[0,1,1,2,2,0],
                &elsup.0, &elsup.1,
                false);
            mshline = delfem3::msh_topology_uniform::mshline_psup(&psup.0, &psup.1);
        }
        let array_edge = numpy::ndarray::Array2::from_shape_vec(
            (mshline.len()/2,2), mshline).unwrap();
        array_edge.into_pyarray(py)
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



    Ok(())
}