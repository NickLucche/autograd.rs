use super::tensor::Tensor;
use ndarray::{ArrayD, ScalarOperand};
use num_traits::{cast::FromPrimitive, float::Float};
use std::ops;
use std::ops::{AddAssign, Sub, Neg, SubAssign, Mul, Div, MulAssign, DivAssign};

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
impl<T> AddAssign<&Tensor<T>> for Tensor<T>
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

impl<T> Neg for Tensor<T> where
    T: Float + FromPrimitive
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // copies data array, ignores grad
        Tensor::from(-self.data().to_owned())
    }
}
impl<T> Neg for &Tensor<T> where
    T: Float + FromPrimitive
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // copies data array, ignores grad
        Tensor::from(-self.data().to_owned())
    }
}

impl<T> Sub<&Tensor<T>> for Tensor<T>
    where
        T: Float + FromPrimitive,
{
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        self + (-rhs)
    }
}
impl<T> Sub<Tensor<T>> for Tensor<T>
    where
        T: Float + FromPrimitive,
{
    type Output = Tensor<T>;
    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        self + (-rhs)
    }
}
impl<T> Sub<&Tensor<T>> for &Tensor<T>
    where
        T: Float + FromPrimitive,
{
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        // avoid creating two new arrays as you would with `self + &(-rhs)`, instead re-use `-rhs`
        let a = &*self.data();
        let t = -rhs;
        t.data_mut().zip_mut_with(a, move |y, &x| *y = x + *y);
        t
    }
}

impl<T> SubAssign<&Tensor<T>> for &Tensor<T> where T: Float+FromPrimitive, ArrayD<T>: for<'a> SubAssign<&'a ArrayD<T>>,{
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        let mut a = self.data_mut();
        let b = &*rhs.data();
        *a -= b;
    }
}

impl<T: Float+FromPrimitive> PartialEq<Tensor<T>> for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        *self.data() == *other.data()
    }
    fn ne(&self, other: &Tensor<T>) -> bool {
        *self.data() != *other.data()
    }
}

impl<T> Mul<T> for Tensor<T> where
T: Float+FromPrimitive+ScalarOperand{
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() * rhs)
    }
}impl<T> Mul<T> for &Tensor<T> where
T: Float+FromPrimitive+ScalarOperand{
    type Output = Tensor<T>;
    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() * rhs)
    }
}
impl<T> MulAssign<T> for Tensor<T> where
T: Float+FromPrimitive+ScalarOperand+MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        let mut a= self.data_mut();
        *a *= rhs;
    }
}
impl<T> Div<T> for Tensor<T> where
T: Float+FromPrimitive+ScalarOperand{
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() / rhs)
    }
}impl<T> Div<T> for &Tensor<T> where
T: Float+FromPrimitive+ScalarOperand{
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        Tensor::from(self.data().to_owned() / rhs)
    }
}
impl<T> DivAssign<T> for Tensor<T> where
    T: Float+FromPrimitive+ScalarOperand+DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        let mut a= self.data_mut();
        *a /= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_adds(){
        let mut a = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])));    
        let b = Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 2.0);
        
        // no copy adds
        a = a + &b;
        assert!(a==Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 3.0));
        let c: Tensor<f64> = a + b;
        assert!(c==Tensor::from(ArrayD::<f64>::ones(IxDyn(&[1, 10])) * 5.0));
        // copy same memory, c mutated as well
        let mut d = c.clone();
        d.fill(7.0);
        assert!(c==d);
        // new tensor
        let new_t = &c + &d;
        assert!(new_t !=c && c==d);
    }
    #[test]
    fn test_scalar_ops(){
        let mut a = Tensor::from(ArrayD::<f32>::ones(IxDyn(&[1, 10])));
        let b = Tensor::from(ArrayD::<f32>::ones(IxDyn(&[1, 10])) * 2.0);

        assert!(&a*2.0==b);
        assert!(a==b/2.0);
        let c = &a * 2.0;
        assert!(&a!=&c);
        a *= 3.0;
        assert!(a==(Tensor::<f32>::ones(&[1,10]) * 3.0));
    }
}