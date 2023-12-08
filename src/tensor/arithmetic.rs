use super::tensor::Tensor;
use crate::operators::operators::shared_ptr_new;
use ndarray::{Array, ArrayD, Dimension};
use num_traits::{cast::FromPrimitive, float::Float};
use std::ops;
use std::ops::{AddAssign, Sub};

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
    T: Float + FromPrimitive,
{
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        let res = &*self.data() + &*rhs.data();
        Tensor::from(res)
    }
}

// A + B
impl<T> ops::Add<Tensor<T>> for Tensor<T>
where
    T: Float + FromPrimitive,
{
    type Output = Tensor<T>;
    fn add(mut self, rhs: Tensor<T>) -> Tensor<T> {
        {
            let mut a = self.data.borrow_mut();
            let b = rhs.data.borrow();
            a.zip_mut_with(&b, move |y, &x| *y = *y + x);
        }
        self
    }
}

// A += &B
impl<T> ops::AddAssign<&Tensor<T>> for Tensor<T>
where
    T: Float + FromPrimitive,
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        let mut a = self.data_mut();
        let b = &*rhs.data();
        *a += b;
    }
}

// A + &B, add(&B) for A
impl<T> ops::Add<&Tensor<T>> for Tensor<T>
where
    T: Float + FromPrimitive,
{
    type Output = Tensor<T>;
    fn add(mut self, rhs: &Tensor<T>) -> Self::Output {
        {
            let mut a = self.data_mut();
            let b = &*rhs.data();
            // NOTE to have op be in-place with same signature, we would need to own the
            // data (not just the tensor). Since we can only own a mut& to data, this is the
            // only in-place operation that ndarray allows us to use to achieve the same in-place result
            // *a += b;
            // syntactically uglier, but should allow us to avoid the AddAssign trait
            a.zip_mut_with(b, move |y, &x| *y = *y + x);
        }
        self
    }
}

// impl<T> ops::AddAssign<&Tensor<T>> for &Tensor<T> where T: Float+FromPrimitive{
//     fn add_assign(&mut self, rhs: &Tensor<T>) {
//         self.data = self.data + &rhs.data;
//     }
// }
