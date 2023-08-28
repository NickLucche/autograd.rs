use tensor::tensor::Tensor;
use std::ops;

/**
 * Arithmetic rules ndarray
 * 
    &A @ &A which produces a new Array
    B @ A which consumes B, updates it with the result, and returns it
    B @ &A which consumes B, updates it with the result, and returns it
    C @= &A which performs an arithmetic operation in place
 */
// A + &B
impl<T, D> ops::Add<Tensor<T, D>> for &Tensor<T, D> {
    fn add(self, rhs: Tensor<T, D>) -> Self::Output {
        
    }
}
impl<T, D> ops::Add<&Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn add(&self, rhs: &Tensor<T, D>) -> Self::Output {
        self.data += rhs.data;
        *self
    }

}

// A + B
impl<T, D> ops::Add<Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(mut self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
        self
    }
}
