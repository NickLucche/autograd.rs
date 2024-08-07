/* Here we define direct operators for Tensor types; initially this would just forward to ndarray,
** handling the RcRefCell wrapper. With the addition of cuda tensors, there's a new layer in between
** so ideally operator should traverse the following: Tensor->StorageType->RcRefCell->ArrayD | CudaArray.
** In practice, we don't always have an impl for each abstraction layer but rather one function at the Tensor
** level that implements the dispatching down to the most concrete type (as of now).
*/

use crate::tensor::Primitive;
use super::tensor::Tensor;
use super::storage::StorageType;
use super::utils::{tensor_op, tensor_op_mut};
use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::Signed;
use std::ops::{self, Not};
use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
impl<T> ops::Add<&Tensor<T>> for &Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        tensor_op(&self, rhs, |a, b| {
            let res = a + b;
            Tensor::from(res)
        })
    }
}

// A + B
impl<T> ops::Add<Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn add(mut self, rhs: Tensor<T>) -> Tensor<T> {
        // {
        //     let mut a = self.data.borrow_mut();
        //     let b = rhs.data.borrow();
        //     a.zip_mut_with(&b, move |y, &x| *y = *y + x);
        // }
        // self
        tensor_op_mut(&mut self, &rhs, |a, b| {
            a.zip_mut_with(b, move |y, &x| *y = *y + x)
        });
        self
    }
}

// A += &B
impl<T> AddAssign<&Tensor<T>> for Tensor<T>
where
    T: Primitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        // let mut a = self.data_mut();
        // let b = &*rhs.data();
        // *a += b;

        tensor_op_mut(self, &rhs, |a, b| *a += b);
    }
}

// A + &B, add(&B) for A
impl<T> ops::Add<&Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn add(mut self, rhs: &Tensor<T>) -> Self::Output {
        // {
        //     let mut a = self.data_mut();
        //     let b = &*rhs.data();
        //     // NOTE to have op be in-place with same signature, we would need to own the
        //     // data (not just the tensor). Since we can only own a mut& to data, this is the
        //     // only in-place operation that ndarray allows us to use to achieve the same in-place result
        //     // *a += b;
        //     // syntactically uglier, but should allow us to avoid the AddAssign trait
        //     a.zip_mut_with(b, move |y, &x| *y = *y + x);
        // }
        tensor_op_mut(&mut self, &rhs, |a, b| {
            a.zip_mut_with(b, move |y, &x| *y = *y + x)
        });
        self
    }
}

impl<T> Neg for Tensor<T>
where
    T: Primitive + Signed,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // copies data array, ignores grad
        Tensor::from(-self.data().to_owned())
    }
}

impl<T> Neg for &Tensor<T>
where
    T: Primitive + Signed,
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // copies data array, ignores grad
        Tensor::from(-self.data().to_owned())
    }
}

impl<T> Sub<&Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn sub(mut self, rhs: &Tensor<T>) -> Self::Output {
        // {
        //     let mut a = self.data.borrow_mut();
        //     let b = rhs.data.borrow();
        //     a.zip_mut_with(&b, move |y, &x| *y = *y - x);
        // }
        tensor_op_mut(&mut self, rhs, |a, b| {
            a.zip_mut_with(&b, move |y, &x| *y = *y - x)
        });
        self
    }
}


impl<T> Sub<&Tensor<T>> for &Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        let a = &*self.data();
        let b = &*rhs.data();
        Tensor::from(a - b)
    }
}

impl<T> Sub<Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        self - &rhs
    }
}


// TODO not sure if needed
// &A -= &B
// impl<T> SubAssign<&Tensor<T>> for &mut Tensor<T>
// where
//     T: Primitive,
//     ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,
// {
//     fn sub_assign(&mut self, rhs: &Tensor<T>) {
//         let mut a = &mut *self.data_mut();
//         let b = &*rhs.data();
//         a -= b;
//     }
// }

// A -= &B
impl<T> SubAssign<&Tensor<T>> for Tensor<T>
where
    T: Primitive,
    // ArrayD<T>: SubAssign<ArrayD<T>>
    ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,
{
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        tensor_op_mut(self, &rhs, |a, b| *a -= b);
    }
}

impl<T: Primitive> PartialEq<Tensor<T>> for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        *self.data() == *other.data()
    }
    fn ne(&self, other: &Tensor<T>) -> bool {
        *self.data() != *other.data()
    }
}

// elementwise-multiplication
impl<T> Mul<&Tensor<T>> for &Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        let res = &*self.data() * &*rhs.data();
        Tensor::from(res)
    }
}

impl<T> Mul<&Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        // NOTE hack, we need to own the data to perform the move op (A*&B), but the only way I found is to get the value
        // out of the RcRefCell wrapper and then put it back in
        let a = self.move_out_data();
        let b = &*rhs.data.borrow();

        let storage =  a * b;
        _ = self.data.replace(storage);
        self
        // this does not work when comparing old and new array address in test_mul, without counting the fact that we recreate 
        // a refcell and lose the grad data 
        // Tensor::from(storage)
        
         
        // old (again) hacky way to modify the data in place without requiring a movable object, but requiring "mut self", so semantically wrong 
        // tensor_op_mut(&mut self, &rhs, |a, b| {
        //     a.zip_mut_with(b, move |y, &x| *y = *y * x)
        // });
        // self
    }
}

impl<T> Mul<Tensor<T>> for Tensor<T>
where
    T: Primitive,
{
    type Output = Tensor<T>;
    fn mul(self, rhs: Tensor<T>) -> Self::Output {
        self * &rhs 
    }
}

// scalar operations
impl<T> Mul<T> for Tensor<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() * rhs)
    }
}

impl<T> Mul<T> for &Tensor<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() * rhs)
    }
}

impl<T> MulAssign<T> for Tensor<T>
where
    T: Primitive + ScalarOperand + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        let mut a = self.data_mut();
        *a *= rhs;
    }
}

impl<T> Div<T> for Tensor<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        Tensor::from(&*self.data() / rhs)
    }
}

impl<T> Div<T> for &Tensor<T>
where
    T: Primitive + ScalarOperand,
{
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        Tensor::from(&*self.data() / rhs)
    }
}

impl<T> DivAssign<T> for Tensor<T>
where
    T: Primitive + ScalarOperand + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        let mut a = self.data_mut();
        *a /= rhs;
    }
}

// Scalar - Tensor
impl Sub<Tensor<f32>> for f32 {
    type Output = Tensor<f32>;
    fn sub(self, rhs: Tensor<f32>) -> Self::Output {
        Tensor::from(self - &*rhs.data())
    }
}

impl Sub<&Tensor<f32>> for f32 {
    type Output = Tensor<f32>;
    fn sub(self, rhs: &Tensor<f32>) -> Self::Output {
        Tensor::from(self - &*rhs.data())
    }
}
// FIXME remove T: Float trait to implement this and properly support ints
// impl Sub<Tensor<i32>> for i32

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use std::ptr;

    #[test]
    fn test_adds() {
        let mut a = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])));
        let b = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 2.0);

        // no copy adds
        a = a + &b;
        assert!(a == Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 3.0));
        let c: Tensor<f64> = a + b;
        assert!(c == Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 5.0));
        // copy same memory, c mutated as well
        let mut d = c.clone();
        d.fill(7.0);
        assert!(c == d);
        // new tensor
        let new_t = &c + &d;
        assert!(new_t != c && c == d);
    }
    #[test]
    fn test_subs() {
        let mut a = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])));
        let b = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 2.0);

        // no copy subs
        a = a - &b;
        assert!(a == Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * -1.0));
        let c: Tensor<f64> = a - b;
        assert!(c == Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * -3.0));
        // copy same memory, c mutated as well
        let mut d = c.clone();
        d.fill(7.0);
        assert!(c == d);
        // new tensor
        let new_t = &c - &d;
        assert!(new_t != c && c == d);
    }

    #[test]
    fn test_scalar_ops() {
        let mut a = Tensor::from(ArrayD::<f32>::ones(IxDyn(&[1, 10])));
        let b = Tensor::from(ArrayD::<f32>::ones(IxDyn(&[1, 10])) * 2.0);

        assert!(&a * 2.0 == b);
        assert!(a == b / 2.0);
        let c = &a * 2.0;
        assert!(&a != &c);
        a *= 3.0;
        assert!(a == (Tensor::<f32>::ones(&[1, 10]) * 3.0));

        let b = 3.0 - &a;
        assert!(b == (Tensor::<f32>::zeros(&[1, 10])));
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from(array![[1., 2.], [3., 4.]]);
        let b = Tensor::from(ArrayD::<f32>::ones(IxDyn(&[2, 2])) * 2.0);
        let c = &a * &b; // new tensor altogheter
        assert!(&a * 2.0 == c);
        let aclone: Tensor<f32> = a.clone(); // same ptr to data, just shell is copied

        // NOTE this still counts as borrowing a! hence it is counted by Ref
        // let adata = &*aclone.data();
        let cdata = &*c.data();
        // ptr comparisons are a bit ugly rn :(
        if let (StorageType::ArrayData(carr), StorageType::ArrayData(aarr)) = (cdata, &*aclone.data()) {
            // let adata = &*aclone.data();

            // mul validity
            assert!(carr == aview2(&[[2., 4.], [6., 8.]]).into_dyn());

            let c_array_addr = carr as *const ArrayD<f32>;
            // this borrows a's array, triggering runtime Rf check
            let a_array_addr = aarr as *const ArrayD<f32>;
            // mul with copy
            assert!(a_array_addr != c_array_addr);
        }

        let c = a * &b; // consume a, keep storage
        let cdata = &*c.data();

        let adata = &*aclone.data();
        if let (StorageType::ArrayData(carr), StorageType::ArrayData(aarr)) = (cdata, adata) {
            let new_array_addr = carr as *const ArrayD<f32>;
            let a_array_addr = aarr as *const ArrayD<f32>;

            // no copy mul
            assert!(a_array_addr == new_array_addr);
        }
    }
}
