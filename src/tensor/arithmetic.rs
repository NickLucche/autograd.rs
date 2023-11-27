use super::tensor::Tensor;
use ndarray::{Array, Dimension};
use num_traits::{cast::FromPrimitive, float::Float};
use std::ops;

/**
 * Arithmetic rules ndarray
 * 
    &A @ &A which produces a new Array
    B @ A which consumes B, updates it with the result, and returns it
    B @ &A which consumes B, updates it with the result, and returns it
    C @= &A which performs an arithmetic operation in place
 */

// &A + &B
impl<T> ops::Add<&Tensor<T>> for &Tensor<T> where T: Float+FromPrimitive{
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        let res = &self.data + &rhs.data;
        Tensor::from(res)
    }

}

// A + B
impl<T> ops::Add<Tensor<T>> for Tensor<T> where T: Float+FromPrimitive{
    type Output = Tensor<T>;
    fn add(mut self, rhs: Tensor<T>) -> Tensor<T> {
        self.data = self.data + &rhs.data;
        self
    }
}

// &A + B, add(B) for &A
// impl<T> ops::Add<Tensor<T>> for &Tensor<T> where T: Float+FromPrimitive{
//     type Output = Tensor<T>;

//     fn add(mut self, rhs: Tensor<T>) -> Self::Output {
//         // FIXME why can I not use this??
//         // self.data += &rhs.data;
//         self.data = self.data + &rhs.data;
//         *self
//     }
// }

// A + &B, add(&B) for A
impl<T> ops::Add<&Tensor<T>> for Tensor<T> where T: Float+FromPrimitive{
    type Output = Tensor<T>;
    fn add(mut self, rhs: &Tensor<T>) -> Self::Output {
        self.data = self.data + &rhs.data;
        self
    }
}

// impl<T> ops::AddAssign<&Tensor<T>> for &Tensor<T> where T: Float+FromPrimitive{
//     fn add_assign(&mut self, rhs: &Tensor<T>) {
//         self.data = self.data + &rhs.data;
//     }
// }
