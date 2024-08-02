use super::tensor::{Primitive, StorageType, CudaData};
use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::Signed;
use std::ops::{self, Not};
use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
impl<T> ops::Add<&StorageType<T>> for &StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;

    fn add(self, rhs: &StorageType<T>) -> Self::Output {
      todo!()
    }
}

// A + B
impl<T> ops::Add<StorageType<T>> for StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;
    fn add(mut self, rhs: StorageType<T>) -> StorageType<T> {
      todo!()
    }
}

// A += &B
impl<T> AddAssign<&StorageType<T>> for StorageType<T>
where
    T: Primitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &StorageType<T>) {
       todo!()
    }
}

// A + &B, add(&B) for A
impl<T> ops::Add<&StorageType<T>> for StorageType<T>
where
    T: Primitive,
{
    type Output = StorageType<T>;
    fn add(mut self, rhs: &StorageType<T>) -> Self::Output {
        todo!()
    }
}


// Subtraction
impl<T> Sub for &StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: Self) -> StorageType<T> {
        todo!()
    }
}

impl<T> Sub for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T> Sub<&StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn sub(self, other: &StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T>SubAssign<&StorageType<T>> for StorageType<T> {
    fn sub_assign(&mut self, other: &StorageType<T>) {
        todo!()
    }
}

// Multiplication
impl<T> Mul for &StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, other: Self) -> StorageType<T> {
        todo!()
    }
}

impl<T> Mul for StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, other: StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T> Mul<&StorageType<T>> for StorageType<T> {
    type Output = StorageType<T>;

    fn mul(self, other: &StorageType<T>) -> StorageType<T> {
        todo!()
    }
}

impl<T> MulAssign<&StorageType<T>> for StorageType<T> {
    fn mul_assign(&mut self, other: &StorageType<T>) {
        todo!()
    }
}

// Negation
impl<T> Neg for StorageType<T> {
    type Output = StorageType<T>;

    fn neg(self) -> StorageType<T> {
        todo!()
    }
}

// Equality comparison
impl<T> PartialEq for StorageType<T> {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl<T> Div<T> for StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T> Div<T> for &StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn div(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T> DivAssign<T> for StorageType<T>
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

impl<T> Mul<T> for StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn mul(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T> Mul<T> for &StorageType<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = StorageType<T>;
    fn mul(self, rhs: T) -> Self::Output {
        todo!()
    }
}

impl<T> MulAssign<T> for StorageType<T>
where
    T: Primitive + ScalarOperand + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
       todo!()
    }
}



/*************** CUDA ops ****************/
// &A + &B
impl<T> ops::Add<&CudaData<T>> for &CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;

    fn add(self, rhs: &CudaData<T>) -> Self::Output {
      todo!()
    }
}

// A + B
impl<T> ops::Add<CudaData<T>> for CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;
    fn add(mut self, rhs: CudaData<T>) -> CudaData<T> {
      todo!()
    }
}

// A += &B
impl<T> AddAssign<&CudaData<T>> for CudaData<T>
where
    T: Primitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &CudaData<T>) {
       todo!()
    }
}

// A + &B, add(&B) for A
impl<T> ops::Add<&CudaData<T>> for CudaData<T>
where
    T: Primitive,
{
    type Output = CudaData<T>;
    fn add(mut self, rhs: &CudaData<T>) -> Self::Output {
        todo!()
    }
}