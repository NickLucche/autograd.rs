use super::Primitive;
use ndarray::{Array, ArrayD, Axis, IxDyn, ArrayView, ArrayViewMut};
use ndarray::iter::{Iter, IterMut};

#[derive(Clone, Debug)]
pub struct CudaData<T> {
    pub ptr: *mut T,
}

impl<T> Drop for CudaData<T> {
    fn drop(&mut self) {
        // deallocate_cuda_memory(self.ptr);  // Replace with your actual deallocation function
    }
}

#[macro_export]
macro_rules! storage_apply {
    ($value:expr, $func_array:expr, $func_cuda:expr) => {
        match $value {
            StorageType::ArrayData(arr) => $func_array(arr),
            StorageType::CudaData(arr) => $func_cuda(arr),
        }
    };
}

#[macro_export]
macro_rules! storage_apply2 {
    ($value1:expr, $value2:expr, $func_array:expr, $func_cuda:expr) => {
        match ($value1, $value2) {
            (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => {
                $func_array(arr_a, arr_b)
            }
            (StorageType::CudaData(cuda_a), StorageType::CudaData(cuda_b)) => {
                $func_cuda(cuda_a, cuda_b)
            }
            _ => panic!("Tensors must be on same device"),
        }
    };
}
// TODO move into own file
#[derive(Clone, Debug)]
pub enum StorageType<T> {
    ArrayData(Array<T, IxDyn>), // CPU
    CudaData(CudaData<T>),      // CUDA
}

impl<T: Primitive> StorageType<T> {
    pub fn ndim(&self) -> usize {
        storage_apply!(&self, |x: &ArrayD<T>| x.ndim(), |x: &CudaData<T>| todo!())
    }
    pub fn shape(&self) -> Vec<usize> {
        storage_apply!(&self, |x: &ArrayD<T>| x.shape().to_vec(), |x: &CudaData<
            T,
        >| todo!())
    }
    pub fn raw_dim(&self) -> Vec<usize> {
        // on raw_dim/shape difference https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.shape
        // all our arrays have dyn shape anyway
        self.shape()
    }
    pub fn is_contiguous(&self) -> bool {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.is_standard_layout(),
            |x: &CudaData<T>| true
        )
    }
    pub fn is_empty(&self) -> bool {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| self.len() == 0,
            |x: &CudaData<T>| self.len() == 0
        )
    }
    pub fn fill(&mut self, el: T) {
        storage_apply!(self, |x: &mut ArrayD<T>| x.fill(el), |x: &mut CudaData<
            T,
        >| todo!())
    }
    pub fn len(&self) -> usize {
        storage_apply!(&self, |x: &ArrayD<T>| x.len(), |x: &CudaData<T>| todo!())
    }

    pub fn t_clone(&self) -> ArrayD<T> {
        storage_apply!(&self, |x: &ArrayD<T>| x.t().to_owned(), |x: &CudaData<
            T,
        >| todo!())
    }

    // ops
    pub fn sum(&self) -> T {
        storage_apply!(&self, |x: &ArrayD<T>| x.sum(), |x: &CudaData<T>| todo!())
    }
    pub fn sum_axis(&self, a: Axis) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.sum_axis(a),
            |x: &CudaData<T>| todo!()
        )
    }
    pub fn mean(&self) -> T {
        storage_apply!(&self, |x: &ArrayD<T>| x.mean().unwrap(), |x: &CudaData<
            T,
        >| todo!())
    }
    pub fn mean_axis(&self, a: Axis) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.mean_axis(a).unwrap(),
            |x: &CudaData<T>| todo!()
        )
    }

    pub fn broadcast(&self, a: IxDyn) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.broadcast(a).unwrap().to_owned(),
            |x: &CudaData<T>| todo!()
        )
    }

    // TODO dont want to implement these for cuda tbh, ndarray dispatch should be handled by caller, hence they just return ndarrays
    pub fn mapv(&self, f: impl Fn(T) -> T) -> ArrayD<T> {
        storage_apply!(&self, |x: &ArrayD<T>| x.mapv(f), |x: &CudaData<T>| todo!())
    }

    pub fn mapv_inplace(&mut self, f: impl Fn(T) -> T) {
        storage_apply!(
            self,
            |x: &mut ArrayD<T>| x.mapv_inplace(f),
            |x: &mut CudaData<T>| todo!()
        )
    }

    pub fn map(&self, f: impl Fn(&T) -> T) -> ArrayD<T> {
        storage_apply!(&self, |x: &ArrayD<T>| x.map(f), |x: &CudaData<T>| todo!())
    }

    pub fn view(&self) -> ArrayView<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.view()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn view_mut(&mut self) -> ArrayViewMut<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.view_mut()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn iter(&self) -> Iter<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.iter()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.iter_mut()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn into_raw_vec(&self) -> Vec<T> {
        if let StorageType::ArrayData(arr) = self {
            arr.clone().into_raw_vec()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }
}