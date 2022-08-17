use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
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

    Ok(())
}