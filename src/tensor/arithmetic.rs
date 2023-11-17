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
impl<T, D> ops::Add<&Tensor<T, D>> for &Tensor<T, D> where T: Float+FromPrimitive, D: Dimension{
    type Output = Tensor<T, D>;

    fn add(self, rhs: &Tensor<T, D>) -> Self::Output {
        let res = &self.data + &rhs.data;
        Tensor::from(res)
    }

}

// A + B
impl<T, D> ops::Add<Tensor<T, D>> for Tensor<T, D> where T: Float+FromPrimitive, D: Dimension{
    type Output = Tensor<T, D>;
    fn add(mut self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self.data = self.data + &rhs.data;
        self
    }
}

// &A + B, add(B) for &A
// impl<T, D> ops::Add<Tensor<T, D>> for &Tensor<T, D> where T: Float+FromPrimitive, D: Dimension{
//     type Output = Tensor<T, D>;

//     fn add(mut self, rhs: Tensor<T, D>) -> Self::Output {
//         // FIXME why can I not use this??
//         // self.data += &rhs.data;
//         self.data = self.data + &rhs.data;
//         *self
//     }
// }

// A + &B, add(&B) for A
impl<T, D> ops::Add<&Tensor<T, D>> for Tensor<T, D> where T: Float+FromPrimitive, D: Dimension{
    type Output = Tensor<T, D>;
    fn add(mut self, rhs: &Tensor<T, D>) -> Self::Output {
        self.data = self.data + &rhs.data;
        self
    }
}

// impl<T, D> ops::AddAssign<&Tensor<T, D>> for &Tensor<T, D> where T: Float+FromPrimitive, D: Dimension{
//     fn add_assign(&mut self, rhs: &Tensor<T, D>) {
//         self.data = self.data + &rhs.data;
//     }
// }
