/// Handling basic CUDA operations here through the Rust runtime interface; more complex 
/// operations as well as operator kernels are at the C++ level
pub mod bindings;
mod tests;

use crate::tensor::Primitive;
use cuda_runtime_sys::{cudaFree, cudaMalloc, cudaMemset};

/// Rust-managed struct to hold a pointer to CUDA memory, (cuda)freed when object is dropped
#[derive(Clone, Debug)]
pub struct CudaData<T: Primitive> {
    ptr: *mut T,

    /// Len in bytes
    size: usize,
    shape: Vec<usize>,
}

impl<T: Primitive> Drop for CudaData<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr.cast());
            }
        }
    }
}

impl<T: Primitive> CudaData<T> {
    // pub fn new(ptr: *mut T, size: usize) -> Self {
    // CudaData { ptr, size }
    // }

    pub fn data(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn data_mut(&mut self) -> *mut T {
        self.ptr
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    // TODO error handling
    fn malloc(size: usize, shape: Option<Vec<usize>>) -> CudaData<T> {
        // reference and kudos to https://github.com/MWATelescope/mwa_hyperdrive/blob/main/src/gpu/mod.rs#L209C1-L223C6
        let mut d_ptr = std::ptr::null_mut();
        unsafe {
            // TODO mallocManaged?
            cudaMalloc(&mut d_ptr, size);
            // check_for_errors(GpuCall::Malloc)?;
        }
        Self {
            ptr: d_ptr.cast(),
            size: size,
            shape: shape.unwrap_or(vec![size]),
        }
    }
}

// creational methods
impl<T: Primitive> CudaData<T> {
    pub fn empty(shape: Vec<usize>) -> CudaData<T> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        Self::malloc(size, Some(shape))
    }

    pub fn zeros(shape: Vec<usize>) -> CudaData<T> {
        let data = Self::empty(shape);
        unsafe  {
            let err = cudaMemset(data.ptr.cast(), 0, data.size); 
        }
        data
    }

    pub fn ones(shape: Vec<usize>) -> CudaData<T> {
        let data = Self::empty(shape);
        unsafe  {
            let err = cudaMemset(data.ptr.cast(), 1, data.size); 
        }
        data
    }
}
