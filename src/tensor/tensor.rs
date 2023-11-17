use crate::autograd::autograd::{backward_algo, Node};
use crate::operators::operators::shared_ptr_new;
use ndarray::linalg::Dot;
use ndarray::{Array, Dimension};
use std::cell::RefCell;
use std::convert::From;
use std::ops;
use std::rc::Rc;
extern crate num_traits;
// use num_traits::Num;
use super::arithmetic::*;
use num_traits::{cast::FromPrimitive, float::Float};

// trait WellBehavedArray<T, D> where T: Float+FromPrimitive, D: Dimension, Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>> {}
// trait WellBehavedArray: PartialOrd + Display {}
// impl<T: PartialOrd + Display> PartialDisplay for T {}

type SharedPtr<T> = Rc<RefCell<T>>;
// we don't want to support all types, we require T to be a number
pub struct Tensor<T: Float + FromPrimitive, D: Dimension> {
    pub requires_grad: bool,
    // TODO rename to graph_node
    pub graph: Option<SharedPtr<Node<T, D>>>,
    // graph: Option< Rc<Node> >,
    pub data: Array<T, D>,
    pub grad: Option<Array<f32, D>>,
    name: String, // for later use if we want to enforce arg order in ops
}
// TODO a .no_grad() can skip graph creation; but by default grad is init lazily
impl<T: Float + FromPrimitive, D: Dimension> From<Array<T, D>> for Tensor<T, D> {
    fn from(arr: Array<T, D>) -> Self {
        Self {
            graph: None,
            data: arr,
            grad: None, // lazy init
            name: "".to_string(),
            requires_grad: true,
        }
    }
}

impl<T, D> Tensor<T, D>
where
    T: Float + FromPrimitive,
    D: Dimension,
{
    pub fn clone(&self) -> Tensor<T, D>
    where
        T: Clone,
    {
        Tensor {
            graph: None,
            data: self.data.clone(),
            grad: self.grad.clone(),
            name: self.name.clone(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = Some(Array::zeros(self.data.raw_dim()));
    }

    pub fn accumulate_grad<A: Float + FromPrimitive>(&mut self, b: &Tensor<A, D>) {
        // if self has no grad, lazy init it here
        match (&mut self.grad, &b.grad) {
            (None, Some(b_grad)) => self.grad = Some(b_grad.clone()),
            (Some(a_grad), Some(b_grad)) => *a_grad += b_grad,
            (_, None) => {}
        }
    }

    pub fn backward(&self) {
        if !self.requires_grad {
            unimplemented!();
        }
        // grad accumulator is always the same size as the output var from which backward is called on!
        // e.g Loss -> self is a "scalar", prev_grad=1
        match &self.graph {
            Some(g) => backward_algo(Rc::clone(g), shared_ptr_new(ones_like_f32(self))),
            _ => panic!("Variable has no attached graph"),
        }
    }
}
impl<T, D> Tensor<T, D>
where
    T: Float + FromPrimitive,
    D: Dimension,
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
{
    pub fn dot(&self, other: &Tensor<T, D>) -> Tensor<T, D> {
        // NOTE this actually only works with 1D/2D matrices! https://docs.rs/ndarray/latest/ndarray/linalg/trait.Dot.html
        let res = self.data.dot(&other.data);
        Tensor::from(res)
    }
}

// TODO other file for ops
pub fn ones_like<T: Float + FromPrimitive, D: Dimension>(t: &Tensor<T, D>) -> Tensor<T, D> {
    let data = Array::<T, D>::ones(t.data.raw_dim());
    Tensor::from(data)
}
pub fn ones_like_f32<T: Float + FromPrimitive, D: Dimension>(t: &Tensor<T, D>) -> Tensor<f32, D> {
    let data = Array::<f32, D>::ones(t.data.raw_dim());
    Tensor::from(data)
}

pub fn zeros_like<T: Float + FromPrimitive, D: Dimension>(t: &Tensor<T, D>) -> Tensor<T, D> {
    let data = Array::<T, D>::zeros(t.data.raw_dim());
    Tensor::from(data)
}
// impl<T, D> From<Array<T, D>> for Tensor<T, D> {
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
        assert_eq!(t.data.ndim(), 2);
        assert_eq!(t.data.len(), 6);
        assert_eq!(t.data.shape(), [2, 3]);
        assert_eq!(t.data.is_empty(), false);
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
