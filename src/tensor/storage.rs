use crate::{storage_apply, storage_apply2};

use super::tensor::{CudaData, Primitive, StorageType};
use super::utils::storage_binary_op;
use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::Signed;
use std::ops::{self, Not};
use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::ops::{Index, IndexMut};

// Tensor + Tensor => StorageType + StorageType  => Array + Array
//                                              \=> CudaData + CudaData

/**
* Arithmetic rules ndarray
*
   &A @ &A which produces a new Array
   B @ A which consumes B, updates it with the result, and returns it
   B @ &A which consumes B, updates it with the result, and returns it
   C @= &A which performs an arithmetic operation in place

    use ndarray::{array, ArrayView1};

    let owned1 = array![1, 2];
    let owned2 = array![3, 4];
    let view1 = ArrayView1::from(&[5, 6]);
    let view2 = ArrayView1::from(&[7, 8]);
    let mut mutable = array![9, 10];

    let sum1 = &view1 + &view2;   // Allocates a new array. Note the explicit `&`.
    // let sum2 = view1 + &view2; // This doesn't work because `view1` is not an owned array.
    let sum3 = owned1 + view1;    // Consumes `owned1` (moves it, need to re-assign it), updates it, and returns it.
    let sum4 = owned2 + &view2;   // Consumes `owned2`, updates it, and returns it.
    mutable += &view2;            // Updates `mutable` in-place.
 */

// &A + &B
impl<T: Primitive> ops::Add<&StorageType<T>> for &StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;

    fn add(self, rhs: &StorageType<T>) -> Self::Output {
        todo!()
    }
}

// A + B
impl<T: Primitive> ops::Add<StorageType<T>> for StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;
    fn add(mut self, rhs: StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

// A += &B
impl<T: Primitive> AddAssign<&StorageType<T>> for StorageType<T>
where
    T: Primitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &StorageType<T>) {
        todo!()
    }
}

// A + &B, add(&B) for A
impl<T: Primitive> ops::Add<&StorageType<T>> for StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;
    fn add(mut self, rhs: &StorageType<T>) -> Self::Output {
        todo!()
    }
}

// Subtraction
impl<T: Primitive> Sub for &StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: Self) -> StorageType<T> {
        todo!()
    }
}

impl<T: Primitive> Sub for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T: Primitive> Sub<&StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: &StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T: Primitive> SubAssign<&StorageType<T>> for StorageType<T> {
    fn sub_assign(&mut self, other: &StorageType<T>) {
        todo!()
    }
}

// Multiplication
// &A * &B
impl<T: Primitive> Mul<&StorageType<T>> for &StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, other: &StorageType<T>) -> StorageType<T> {
        storage_binary_op(self, other, |a, b| StorageType::ArrayData(a * b))
    }
}

// A * &B
impl<T: Primitive> Mul<&StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, other: &StorageType<T>) -> StorageType<T> {
        // TODO factor out the function with move 
        match (self, other) {
            (StorageType::ArrayData(mut arr_a), StorageType::ArrayData(arr_b)) => {
                arr_a = arr_a * arr_b;
                StorageType::ArrayData(arr_a)
            },
            (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
            _ => panic!("Tensors must be on same device"), // TODO return proper result
        }
    }
}

impl<T: Primitive> Mul<StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, rhs: StorageType<T>) -> Self::Output {
        // same as above A * &B
        self * &rhs
    }
}

// A * &B
impl<T> MulAssign<T> for StorageType<T>
where
    T: Primitive + ScalarOperand + MulAssign{
    fn mul_assign(&mut self, scalar: T) {
        match self {
            StorageType::ArrayData(arr_a) => *arr_a *= scalar,
            _ => todo!(),
        }        
    }
}

// Negation
impl<T: Primitive> Neg for StorageType<T> {
    type Output = StorageType<T>;

    fn neg(self) -> StorageType<T> {
        todo!()
    }
}

// Equality comparison
impl<T: Primitive> PartialEq<StorageType<T>> for StorageType<T> {
    fn eq(&self, other: &StorageType<T>) -> bool {
        storage_apply2!(
            self,
            other,
            |a: &ArrayD<T>, b: &ArrayD<T>| a == b,
            |a: &CudaData<T>, b: &CudaData<T>| todo!()
        )
    }
    fn ne(&self, other: &StorageType<T>) -> bool {
        storage_apply2!(
            self,
            other,
            |a: &ArrayD<T>, b: &ArrayD<T>| a != b,
            |a: &CudaData<T>, b: &CudaData<T>| todo!()
        )
    }
}
impl<T: Primitive> PartialEq<ArrayD<T>> for &StorageType<T> {
    fn eq(&self, other: &ArrayD<T>) -> bool {
        if let StorageType::ArrayData(arr) = &self {
            return arr == other;
        } else {
            todo!()
        }
    }
    fn ne(&self, other: &ArrayD<T>) -> bool {
        if let StorageType::ArrayData(arr) = &self {
            return arr != other;
        } else {
            todo!()
        }
    }
}

impl<T: Primitive> PartialEq<ArrayD<T>> for StorageType<T> {
    fn eq(&self, other: &ArrayD<T>) -> bool {
        return &self == other;
    }
    fn ne(&self, other: &ArrayD<T>) -> bool {
        return &self != other;
    }
}

impl<T: Primitive> Div<T> for StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T: Primitive> Div<T> for &StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T: Primitive> DivAssign<T> for StorageType<T>
where
    T: Primitive + ScalarOperand + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        todo!()
    }
}

// Scalar - Tensor
impl Sub<StorageType<f32>> for f32 {
    type Output = StorageType<f32>;
    fn sub(self, rhs: StorageType<f32>) -> Self::Output {
        todo!()
    }
}

impl Sub<&StorageType<f32>> for f32 {
    type Output = StorageType<f32>;
    fn sub(self, rhs: &StorageType<f32>) -> Self::Output {
        todo!()
    }
}

// Scalar * A, will create a new storage
impl<T: Primitive> Mul<T> for &StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn mul(self, scalar: T) -> Self::Output {
        storage_apply!(self,
            |a: &ArrayD<T>| StorageType::ArrayData(a * scalar),
            |a: &CudaData<T>| todo!()
        )

    }
}

impl<T: Primitive> Mul<T> for StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn mul(self, scalar: T) -> Self::Output {
        &self * scalar
    }
}


/** Indexing **/
// impl<T: Primitive> StorageType<T> {
//     fn compute_flat_index(&self, index: &[usize]) -> usize {
//         // Compute a flat index from the multi-dimensional index.
//         // This usually involves multiplying the index components by the size of the corresponding dimensions.
//         // Here, we assume row-major order for simplicity.
//         let mut flat_index = 0;
//         for (i, &dim_size) in self.dims.iter().enumerate() {
//             flat_index = flat_index * dim_size + index[i];
//         }
//         flat_index
//     }
// }

impl<T: Primitive> Index<&[usize]> for StorageType<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if let StorageType::ArrayData(arr) = &self {
            return &arr[index];
        } else {
            todo!()
        }
    }
}

impl<T: Primitive> IndexMut<&[usize]> for StorageType<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        if let StorageType::ArrayData(arr) = self {
            return arr.index_mut(index);
        } else {
            todo!()
        }
    }
}

// index with fixed-size arrays
macro_rules! add_indexing {
    ($dims:expr) => {
        impl<T: Primitive> Index<[usize; $dims]> for StorageType<T> {
            type Output = T;

            fn index(&self, index: [usize; $dims]) -> &Self::Output {
                if let StorageType::ArrayData(arr) = &self {
                    return &arr[index];
                } else {
                    todo!()
                }
            }
        }

        impl<T: Primitive> IndexMut<[usize; $dims]> for StorageType<T> {
            fn index_mut(&mut self, index: [usize; $dims]) -> &mut Self::Output {
                if let StorageType::ArrayData(arr) = self {
                    return arr.index_mut(index);
                } else {
                    todo!()
                }
            }
        }
    };
}

add_indexing!(1);
add_indexing!(2);
add_indexing!(3);
add_indexing!(4);
add_indexing!(5);

/*************** CUDA ops ****************/
// &A + &B
impl<T: Primitive> ops::Add<&CudaData<T>> for &CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;

    fn add(self, rhs: &CudaData<T>) -> Self::Output {
        todo!()
    }
}

// A + B
impl<T: Primitive> ops::Add<CudaData<T>> for CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;
    fn add(mut self, rhs: CudaData<T>) -> CudaData<T> {
        todo!()
    }
}

// A += &B
impl<T: Primitive> AddAssign<&CudaData<T>> for CudaData<T>
where
    T: Primitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &CudaData<T>) {
        todo!()
    }
}

// A + &B, add(&B) for A
impl<T: Primitive> ops::Add<&CudaData<T>> for CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;
    fn add(mut self, rhs: &CudaData<T>) -> Self::Output {
        todo!()
    }
}
