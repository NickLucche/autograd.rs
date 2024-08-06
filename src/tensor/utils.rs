use super::tensor::{Primitive, StorageType, Tensor};
use ndarray::{ArrayD, IxDyn, ScalarOperand};

// helper functions for dispatching based on StorageType
pub fn tensor_op<T: Primitive>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    cpu_f: fn(&ArrayD<T>, &ArrayD<T>) -> Tensor<T>,
) -> Tensor<T> {
    let storage = lhs.data();
    let rstorage = rhs.data();
    match (&*storage, &*rstorage) {
        (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => cpu_f(arr_a, arr_b),
        (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
        _ => panic!("Tensors must be on same device"), // TODO return proper result
    }
}

pub fn tensor_op_mut<T: Primitive>(
    lhs: &mut Tensor<T>,
    rhs: &Tensor<T>,
    cpu_f: impl Fn(&mut ArrayD<T>, &ArrayD<T>),
) {
    let mut storage = lhs.data_mut();
    let rstorage = rhs.data();
    match (&mut *storage, &*rstorage) {
        (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => cpu_f(arr_a, arr_b),
        (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
        _ => panic!("Tensors must be on same device"),
    }
}

pub fn storage_binary_op<T: Primitive>(
    lhs: &StorageType<T>,
    rhs: &StorageType<T>,
    cpu_f: fn(&ArrayD<T>, &ArrayD<T>) -> StorageType<T>,
) -> StorageType<T> {
    match (lhs, rhs) {
        (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => cpu_f(arr_a, arr_b),
        (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
        _ => panic!("Tensors must be on same device"), // TODO return proper result
    }
}

pub fn storage_binary_op_mut<T: Primitive>(
    lhs: &mut StorageType<T>,
    rhs: &StorageType<T>,
    cpu_f: impl Fn(&mut ArrayD<T>, &ArrayD<T>) -> StorageType<T>,
) -> StorageType<T> {
    match (lhs, rhs) {
        (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => cpu_f(arr_a, arr_b),
        (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
        _ => panic!("Tensors must be on same device"), // TODO return proper result
    }
}


pub fn storage_binary_move_op<T:Primitive>(a: StorageType<T>, b: &StorageType<T>, cpu_f: impl Fn(ArrayD<T>, &ArrayD<T>)->StorageType<T>)-> StorageType<T> {
    match (a, b) {
        (StorageType::ArrayData(mut arr_a), StorageType::ArrayData(arr_b)) => cpu_f(arr_a, arr_b),
        (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
        _ => panic!("Tensors must be on same device"), // TODO return proper result
    }
}