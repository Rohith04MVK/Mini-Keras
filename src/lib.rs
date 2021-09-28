use pyo3::prelude::*;

// Main pymodule
#[pymodule]
fn _mini_keras(_py: Python, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
