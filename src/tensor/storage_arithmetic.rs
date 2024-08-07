use crate::{storage_apply, storage_apply2};

use crate::tensor::Primitive;
use super::storage::{CudaData, StorageType};
use super::utils::{storage_binary_move_op, storage_binary_op, storage_binary_op_mut};
use ndarray::{ArrayD, ScalarOperand};
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
// A - &B
impl<T: Primitive> Sub<&StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: &StorageType<T>) -> StorageType<T> {
        // respect move semantics here (let c = a - b)
        storage_binary_move_op(self, other, |mut a, b| {
            a = a - b;
            StorageType::ArrayData(a)
        })
    }
}
// &A - B
// leave unimplented for now, doesn't seem necessary
// impl<T: Primitive> Sub<StorageType<T>> for &StorageType<T> {
//     type Output = StorageType<T>;

//     fn sub(self, rhs: StorageType<T>) -> Self::Output {
//         storage_binary_move_op(rhs, self, |a, b| StorageType::ArrayData(a - b))
//     }
// }
// A - B
impl<T: Primitive> Sub<StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: Self) -> StorageType<T> {
        self - &other
    }
}

// &A - &B
impl<T: Primitive> Sub<&StorageType<T>> for &StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: &StorageType<T>) -> Self::Output {
        storage_binary_op(self, other, |a, b| StorageType::ArrayData(a - b))
    }
}



// &A -= &B
// impl<T> SubAssign<&StorageType<T>> for &mut StorageType<T>
// where
//     T: Primitive,
//     ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,
// {
//     fn sub_assign(&mut self, rhs: &StorageType<T>) {
//         match (self, rhs) {
//             (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => *arr_a -= arr_b,
//             (StorageType::CudaData(arr_a), StorageType::CudaData(arr_b)) => todo!(),
//             _ => panic!("Tensors must be on same device"), // TODO return proper result
//         }
//         // storage_apply2!(
//         //     self,
//         //     other,
//         //     |a: &mut ArrayD<T>, b: ArrayD<T>| *a -= b,
//         //     |a: &CudaData<T>, b: &CudaData<T>| todo!()
//         // )
//     }
// }


impl<T> SubAssign<&StorageType<T>> for StorageType<T> 
// where ArrayD<T>: SubAssign<ArrayD<T>>
where T:Primitive, ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,

{
    fn sub_assign(&mut self, other: &StorageType<T>) {
        storage_apply2!(
            self,
            other,
            |a: &mut ArrayD<T>, b: &ArrayD<T>| *a -= b,
            |a: &CudaData<T>, b: &CudaData<T>| todo!()
        )
    }
}

impl<T> SubAssign<StorageType<T>> for StorageType<T> 
where T:Primitive, ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,
{
    fn sub_assign(&mut self, rhs: StorageType<T>) {
        *self -= &rhs;
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
        storage_binary_move_op(self, other, |mut a, b| {
            a = a * b;
            StorageType::ArrayData(a)
        })
    }
}

impl<T: Primitive> Mul<StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, rhs: StorageType<T>) -> Self::Output {
        // same as above A * &B
        self * &rhs
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

//** Scalar operators **/

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

impl<T: Primitive> Div<T> for &StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, scalar: T) -> Self::Output {
        // return new storage
        storage_apply!(self,
            |a: &ArrayD<T>| StorageType::ArrayData(a / scalar),
            |a: &CudaData<T>| todo!()
        )
    }
}

impl<T: Primitive> Div<T> for StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, scalar: T) -> Self::Output {
        &self / scalar
    }
}

impl<T: Primitive> DivAssign<T> for StorageType<T>
where
    T: Primitive + ScalarOperand + DivAssign,
{
    fn div_assign(&mut self, scalar: T) {
        match self {
            StorageType::ArrayData(arr_a) => *arr_a /= scalar,
            _ => todo!(),
        }        
    }
}

// *Scalar* - Tensor

impl Sub<&StorageType<f32>> for f32 {
    type Output = StorageType<f32>;
    fn sub(self, rhs: &StorageType<f32>) -> Self::Output {
        storage_apply!(rhs,
            |a: &ArrayD<f32>| {
                // transform scalar into array and do sub
                let scalar_arr = ndarray::array![self].into_dyn();
                StorageType::ArrayData(scalar_arr - a)
            },
            |a: &CudaData<f32>| todo!()
        )
    }
}

impl Sub<StorageType<f32>> for f32 {
    type Output = StorageType<f32>;
    fn sub(self, rhs: StorageType<f32>) -> Self::Output {
        self - &rhs
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
