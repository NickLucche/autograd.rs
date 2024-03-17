use crate::autograd::autograd::{backward_algo, Node};
use crate::operators::operators::shared_ptr_new;
use ndarray::linalg::Dot;
use ndarray::{array, Array, ArrayD, Axis, Dimension, Ix2, IxDyn};
use std::cell::{Ref, RefCell, RefMut};
use std::convert::From;
use std::ops::AddAssign;
use std::rc::Rc;

extern crate num_traits;

use super::init::{kaiming_uniform, uniform};
use num_traits::{cast::FromPrimitive, Num, NumCast};
pub trait Primitive: Copy + NumCast + Num + PartialOrd<Self> + Clone + FromPrimitive {}
impl Primitive for u8 {}
impl Primitive for f32 {}
impl Primitive for f64 {}
impl Primitive for i32 {}
impl Primitive for i64 {}

// trait WellBehavedArray<T, D> where T: Float+FromPrimitive, D: Dimension, Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>> {}
// trait WellBehavedArray: PartialOrd + Display {}
// impl<T: PartialOrd + Display> PartialDisplay for T {}

// TODO for view+owned, though not elegant https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=4095c89a23a2339a1e7afe3813c9acc3
type SharedPtr<T> = Rc<RefCell<T>>;
type SharedArray<T> = SharedPtr<Array<T, IxDyn>>;

fn deep_copy_shared_array<T: Clone>(t: &SharedArray<T>) -> SharedArray<T> {
    shared_ptr_new(t.borrow().to_owned())
}

fn add_shared_array_inplace<T: Primitive>(a: &mut SharedArray<T>, b: &SharedArray<T>)
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
pub struct Tensor<T: Primitive> {
    pub requires_grad: bool,
    // TODO rename to graph_node
    pub graph: Option<SharedPtr<Node<T>>>,
    pub data: SharedPtr<Array<T, IxDyn>>,
    pub grad: SharedPtr<Option<Array<f32, IxDyn>>>,
    name: String, // for later use if we want to enforce arg order in ops
}

// TODO a .no_grad() to set all to requires_grad to false and NOT create any graph
impl<T: Primitive, D: Dimension> From<Array<T, D>> for Tensor<T> {
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
    T: Primitive,
{
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            graph: None,
            data: shared_ptr_new(ArrayD::<T>::zeros(IxDyn(shape))),
            grad: shared_ptr_new(None),
            name: "".to_string(),
            requires_grad: true,
        }
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            graph: None,
            data: shared_ptr_new(ArrayD::<T>::ones(IxDyn(shape))),
            grad: shared_ptr_new(None),
            name: "".to_string(),
            requires_grad: true,
        }
    }

    pub fn uniform(tensor_shape: &[usize], low: f32, high: f32) -> Tensor<f32> {
        uniform(tensor_shape, low, high)
    }

    pub fn kaiming_uniform(tensor_shape: &[usize]) -> Tensor<f32> {
        kaiming_uniform(tensor_shape)
    }
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

    pub fn accumulate_grad<A: Primitive>(&mut self, b: &Tensor<A>) {
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

    pub fn t(&self) -> &Self {
        assert_eq!(self.ndim(), 2);
        self.swap_axes(0, -1);
        self
    }
    pub fn t_clone(&self) -> Self {
        Tensor::from(self.data().t().to_owned())
    }
    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }
    pub fn swap_axes(&self, mut ax: i32, mut bx: i32) {
        let mut t = self.data_mut();
        if ax < 0 {
            ax = t.ndim() as i32 + ax;
        }
        if bx < 0 {
            bx = t.ndim() as i32 + bx;
        }
        t.swap_axes(ax as usize, bx as usize);
    }

    pub fn sum(&self) -> T {
        self.data().sum()
    }

    pub fn as_type<A: Primitive>(&self) -> Tensor<A> {
        // TODO in-place with cast if possible? https://github.com/rust-ndarray/ndarray/issues/493
        let mut t = Tensor::from(self.data().mapv(|elem| A::from(elem).unwrap()));
        t.requires_grad = self.requires_grad;
        t
    }

    pub fn fill(&mut self, x: T) {
        self.data_mut().fill(x)
    }
    pub fn is_contiguous(&self) -> bool {
        self.data().is_standard_layout()
    }
}

// trait for "specialized" functions
// NOTE use this to add extra functions that have different impls per type
pub trait Powi {
    fn powi(&self, exp: i32) -> Self;
    fn powi_inplace(self, exp: i32) -> Self;
}

// fast exp from Float trait for floats
impl Powi for Tensor<f32> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(self.data().mapv(|a| a.powi(exp)))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.data_mut().mapv_inplace(|a| a.powi(exp));
        self
    }
}
impl Powi for Tensor<f64> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(self.data().mapv(|a| a.powi(exp)))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.data_mut().mapv_inplace(|a| a.powi(exp));
        self
    }
}

// "default" T::pow implementations for other primitives
impl Powi for Tensor<u8> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(self.data().mapv(|a| u8::pow(a, exp as u32)))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.data_mut().mapv_inplace(|a| u8::pow(a, exp as u32));
        self
    }
}

impl Powi for Tensor<i32> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(self.data().mapv(|a| i32::pow(a, exp as u32)))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.data_mut().mapv_inplace(|a| i32::pow(a, exp as u32));
        self
    }
}
impl Powi for Tensor<i64> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(self.data().mapv(|a| i64::pow(a, exp as u32)))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.data_mut().mapv_inplace(|a| i64::pow(a, exp as u32));
        self
    }
}

impl<T> Tensor<T>
where
    T: Primitive + 'static,
    Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
{
    pub fn dot(&self, other: &Tensor<T>) -> Tensor<T> {
        // NOTE this actually only works with 1D/2D matrices! https://docs.rs/ndarray/latest/ndarray/linalg/trait.Dot.html
        if self.data().ndim() > 2 || other.data().ndim() > 2 {
            panic!("Ndarray only supports 1d/2d matmul!");
        }
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
    T: Primitive,
{
    pub fn shape(&self) -> Vec<usize> {
        self.data().shape().to_owned()
    }
    pub fn shapei(&self, i: usize) -> usize {
        self.shape()[i]
    }
    pub fn reshape(&mut self, shape: &[usize]) {
        // we cannot directly modify the stride, so we need another Array which points to the same
        // storage.. thing is we can't move out the array either so we
        // have to work around the move by replacing the Rf value with the reshaped Array, which, internally,
        // is a new ndarray::ArrayBase with new strides but same data pointer; pretty hacky,
        // might want to integrate with a unified Array/View container so we keep a view here and dont
        // change every clone

        // can't move with into_inner/take..
        let array = self.data.replace(ArrayD::<T>::zeros(IxDyn(&[1])));
        // NOTE will panic if the array is *NOT* contiguous
        let reshaped_array = array.into_shape(shape).unwrap();
        let _ = self.data.replace(reshaped_array);
        // similar to
        // let reshaped_array = unsafe {
        //  Array::from_shape_vec_unchecked(shape, data_ptr)
        // };
        // let old = std::mem::replace(&mut *self.data.borrow_mut(), reshaped_array);
    }
    pub fn unsqueeze(mut self, mut ax: i32) -> Self {
        if ax < 0 {
            ax = self.ndim() as i32;
        }
        let ax = ax as usize;
        assert!(ax <= self.ndim()); // does unsqueze(100) make sense?
        let mut shape = self.shape();
        if ax < shape.len() {
            shape.insert(ax, 1);
        } else {
            shape.push(1);
        }
        self.reshape(shape.as_slice());
        self
    }
    pub fn squeeze(mut self) -> Self {
        let mut shape = self.shape();
        shape.retain(|&x| x != 1);
        self.reshape(shape.as_slice());
        self
    }
    pub fn size(&self) -> usize {
        self.len()
    }
    pub fn len(&self) -> usize {
        self.data().len()
    }
    pub fn sum_axis(&self, mut axis: i32) -> Tensor<T> {
        if axis < 0 {
            axis = self.ndim() as i32 + axis;
        }
        Tensor::from(self.data().sum_axis(Axis(axis as usize)))
    }
    pub fn mean(&self, axis: Option<usize>) -> Tensor<T> {
        // will panic on empty tensors, at least it's consistent with .sum
        match axis {
            Some(ax) => Tensor::from(self.data().mean_axis(Axis(ax)).unwrap()),
            None => Tensor::from(array![self.data().mean().unwrap()])
        }
    }
}

// TODO other file for ops
pub fn ones_like<T: Primitive>(t: &Tensor<T>) -> Tensor<T> {
    let data = ArrayD::<T>::ones(t.data().raw_dim());
    Tensor::from(data)
}

pub fn ones_like_f32<T: Primitive>(t: &Tensor<T>) -> Tensor<f32> {
    let data = ArrayD::<f32>::ones(t.data().raw_dim());
    Tensor::from(data)
}

pub fn zeros_like<T: Primitive>(t: &Tensor<T>) -> Tensor<T> {
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
        assert_ne!(ptr::addr_of!(t), ptr::addr_of!(t2));

        let tdata = &*t.data();
        let tp = tdata as *const ArrayD<f64>;
        let t2data = &*t2.data();
        let tp2 = t2data as *const ArrayD<f64>;
        assert!(tp == tp2);
        assert_eq!(*t.data(), *t2.data());
    }
    #[test]
    fn test_reshape() {
        let a = array![[1., 2.], [3., 4.]];
        let t = Tensor::from(a);
        let mut t2 = t.clone();
        t2.reshape(&[4]);
        // all clones of t point to the same storage, which has now been modified
        // with different shape+stride
        assert!(t2.shape() == t.shape() && t.shape() == vec![4]);
    }
}
