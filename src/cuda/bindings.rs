// Rust Foreign Function Interface (FFI) to link against the CUDA shared library
// can be automatically generated using bindgen
use std::os::raw::{c_int, c_float};

#[repr(C)]
pub struct Dim {
    b: c_int,
    h: c_int,
    w: c_int,
    c: c_int,
}


extern "C" {
    // fn testKernelWrapper(a: c_int, b: c_int) -> c_int;
    pub fn testKernelWrapper();
    fn conv2dConcurrent(
        data_in: *mut c_float, 
        kernel: *mut c_float, 
        data_out: *mut c_float, 
        in_dim: Dim, 
        kernel_dim: Dim, 
        stride: Dim, 
        pad: c_int
    ) -> Dim;
}