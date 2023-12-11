use crate::autograd::autograd::{backward_algo, Node};
use crate::operators::operators::shared_ptr_new;
use ndarray::linalg::Dot;
use ndarray::{
    array, Array, ArrayD, ArrayView, ArrayViewMut, Axis, Dimension, Ix2, IxDyn, RemoveAxis,
};
use std::cell::{Ref, RefCell, RefMut};
use std::convert::From;
use std::ops;
use std::ops::AddAssign;
use std::rc::Rc;
extern crate num_traits;
// use num_traits::Num;
use super::arithmetic::*;
use num_traits::{cast::FromPrimitive, float::Float};

// trait WellBehavedArray<T, D> where T: Float+FromPrimitive, D: Dimension, Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>> {}
// trait WellBehavedArray: PartialOrd + Display {}
// impl<T: PartialOrd + Display> PartialDisplay for T {}

// TODO for view+owned, though not elegant https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=4095c89a23a2339a1e7afe3813c9acc3
type SharedPtr<T> = Rc<RefCell<T>>;
type SharedArray<T> = SharedPtr<Array<T, IxDyn>>;
fn deep_copy_shared_array<T: Clone>(t: &SharedArray<T>) -> SharedArray<T> {
    shared_ptr_new(t.borrow().to_owned())
}
fn add_shared_array_inplace<T: Float + FromPrimitive>(a: &mut SharedArray<T>, b: &SharedArray<T>)
where
    ArrayD<T>: for<'a> AddAssign<&'a ArrayD<T>>,
{
    let mut a_t = a.borrow_mut();
    let b_t = &b.borrow();
    *a_t += *&b_t;
}

#[derive(Clone)]
// clone is now inexpensive as we're just referencing the data, not owning it
// we don't want to support all types, we require T to be a number
pub struct Tensor<T: Float + FromPrimitive> {
    pub requires_grad: bool,
    // TODO rename to graph_node
    pub graph: Option<SharedPtr<Node<T>>>,
    pub data: SharedPtr<Array<T, IxDyn>>,
    pub grad: SharedPtr<Option<Array<f32, IxDyn>>>,
    name: String, // for later use if we want to enforce arg order in ops
}
// TODO a .no_grad() can skip graph creation; but by default grad is init lazily
impl<T: Float + FromPrimitive, D: Dimension> From<Array<T, D>> for Tensor<T> {
    fn from(arr: Array<T, D>) -> Self {
        let shape = arr.raw_dim().into_dyn();
        // TODO requires_grad false constructor
        Self {
            graph: None,
            data: shared_ptr_new(arr.into_dyn()),
            grad: shared_ptr_new(None),
            name: "".to_string(), // TODO switch to id?
            requires_grad: true,
        }
    }
}

impl<T> Tensor<T>
where
    T: Float + FromPrimitive,
{
    pub fn to_owned(&self) -> Tensor<T>
    where
        T: Clone,
    {
        // TODO impl trait
        Tensor {
            graph: None,
            data: self.data.to_owned(),
            grad: self.grad.to_owned(),
            name: self.name.to_owned(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn data(&self) -> Ref<ArrayD<T>> {
        self.data.borrow()
    }
    pub fn data_mut(&self) -> RefMut<ArrayD<T>> {
        self.data.borrow_mut()
    }

    pub fn grad(&self) -> Ref<Option<ArrayD<f32>>> {
        self.grad.borrow()
    }
    pub fn grad_mut(&self) -> RefMut<Option<ArrayD<f32>>> {
        self.grad.borrow_mut()
    }

    pub fn zero_grad(&mut self) {
        // need to get this first or I get a simultaneous borrow and mutation of an object error..
        let dim = self.data().raw_dim();
        *self.grad.borrow_mut() = Some(ArrayD::zeros(dim));
    }

    pub fn accumulate_grad<A: Float + FromPrimitive>(&mut self, b: &Tensor<A>) {
        // if self has no grad, lazy init it here
        let mut a_grad = self.grad_mut();
        let b_grad = b.grad();

        match (&mut *a_grad, &*b_grad) {
            (None, Some(b_grad_val)) => *a_grad = Some(b_grad_val.to_owned()),
            (Some(a_grad_val), Some(b_grad_val)) => *a_grad_val += b_grad_val,
            (_, None) => {}
        }
    }

    pub fn accumulate_grad_from_grad_tensor(&mut self, grad_tensor: &Tensor<f32>) {
        // same as above, but we use grad_tensor `.data` and hence assume it's present
        let mut a_grad = self.grad_mut();
        let b_grad = grad_tensor.data();

        match &mut *a_grad {
            None => *a_grad = Some(b_grad.to_owned()),
            Some(a_grad_val) => *a_grad_val += &*b_grad,
        }
    }

    // TODO use view
    // pub fn t_copy(&mut self) -> &Self {
    //     self.data = self.data.t().to_owned();
    //     self
    // }

    pub fn t_clone(&self) -> Self {
        Tensor::from(self.data().t().to_owned())
    }

    pub fn sum(&self) -> T {
        self.data().sum()
    }
    pub fn powi(&self, exp: i32)->Self {
        Tensor::from(self.data().mapv(|a| a.powi(exp)))
    }
    pub fn powi_inplace(self, exp: i32)->Self {
        self.data_mut().mapv_inplace(|a| a.powi(exp));
        self
    }

    pub fn as_type<A: Float+FromPrimitive>(&self)->Tensor<A> {
        // TODO in-place with cast if possible? https://github.com/rust-ndarray/ndarray/issues/493
        Tensor::from(self.data().mapv(|elem| A::from(elem).unwrap()))
    }

    pub fn fill(&mut self, x: T) {
        self.data_mut().fill(x)
    }

}
impl<T> Tensor<T>
where
    T: Float + FromPrimitive + 'static,
    Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
{
    pub fn dot(&self, other: &Tensor<T>) -> Tensor<T> {
        // NOTE this actually only works with 1D/2D matrices! https://docs.rs/ndarray/latest/ndarray/linalg/trait.Dot.html
        // no clone version, operating on views; needs let binding to create a longer lived value..
        let a_ref = self.data();
        let a = a_ref.view().into_dimensionality::<Ix2>().unwrap();
        let b_ref = other.data();
        let b = b_ref.view().into_dimensionality::<Ix2>().unwrap();
        let res = a.dot(&b);
        Tensor::from(res)
    }
}

trait Backward {
    fn do_backward(&self);
}
impl Backward for Tensor<f32> {
    fn do_backward(&self) {
        if !self.requires_grad {
            unimplemented!();
        }
        // grad accumulator is always the same size as the output var from which backward is called on!
        // e.g Loss -> self is a "scalar", prev_grad=1
        match &self.graph {
            // these grad accumulator tensors don't need .grad, hence its value remains None!
            Some(g) => backward_algo(Rc::clone(g), ones_like_f32(self)),
            _ => panic!("Variable has no attached graph"),
        }
    }
}
// this is how you would support f32 or f64 backward call
// impl backwardTrait for Tensor<f64> {
//     fn do_backward(&self) {
//
//     }
// }

impl Tensor<f32>
where
    Array<f32, Ix2>: Dot<Array<f32, Ix2>, Output = Array<f32, Ix2>>,
{
    /**
     * Backward is only implemented for f32 tensors as the whole backward pass is run @floating point precision.
     */
    pub fn backward(&self) {
        self.do_backward()
    }
}

impl<T> Tensor<T>
where
    T: Float + FromPrimitive,
{
    pub fn sum_axis(&self, axis: usize) -> Tensor<T> {
        Tensor::from(self.data().sum_axis(Axis(axis)))
    }
}

// TODO other file for ops
pub fn ones_like<T: Float + FromPrimitive>(t: &Tensor<T>) -> Tensor<T> {
    let data = ArrayD::<T>::ones(t.data().raw_dim());
    Tensor::from(data)
}
pub fn ones_like_f32<T: Float + FromPrimitive>(t: &Tensor<T>) -> Tensor<f32> {
    let data = ArrayD::<f32>::ones(t.data().raw_dim());
    Tensor::from(data)
}

pub fn zeros_like<T: Float + FromPrimitive>(t: &Tensor<T>) -> Tensor<T> {
    let data = ArrayD::<T>::zeros(t.data().raw_dim());
    Tensor::from(data)
}
// impl<T, D> From<Array<T, D>> for Tensor<T> {
//     fn from(value: Array<T, D>) -> Self {
//         Tensor { requires_grad: false, graph: None, data: value, grad: array![[]], name: "".to_string() }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use std::ptr;

    #[test]
    fn test_basic_from() {
        let a = Array::range(0., 6., 1.).into_shape([2, 3]).unwrap();
        let t = Tensor::from(a);
        assert_eq!(t.data().ndim(), 2);
        assert_eq!(t.data().len(), 6);
        assert_eq!(t.data().shape(), [2, 3]);
        assert_eq!(t.data().is_empty(), false);
    }

    #[test]
    fn test_clone_tensor() {
        let a = array![[1., 2.], [3., 4.]];
        let t = Tensor::from(a);
        let t2 = t.clone();
        println!("{:?}, {:?}", ptr::addr_of!(t), ptr::addr_of!(t2));
        assert_ne!(ptr::addr_of!(t), ptr::addr_of!(t2));
    }
}
