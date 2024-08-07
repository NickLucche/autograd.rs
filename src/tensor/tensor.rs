use crate::autograd::autograd::{backward_algo, Node};
use crate::utils::shared_ptr_new;
use ndarray::linalg::Dot;
use ndarray::{array, Array, ArrayD, Axis, Dimension, Ix2, IxDyn};
use std::cell::{Ref, RefCell, RefMut};
use std::convert::From;
use std::ops::AddAssign;
use std::rc::Rc;

extern crate num_traits;
// TODO move tensor to mod.rs?
use super::init::{kaiming_uniform, uniform};
use super::storage::StorageType;
use super::Primitive;
use crate::{storage_apply, storage_apply2};

// trait WellBehavedArray<T, D> where T: Float+FromPrimitive, D: Dimension, Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>> {}
// trait WellBehavedArray: PartialOrd + Display {}
// impl<T: PartialOrd + Display> PartialDisplay for T {}

// TODO for view+owned, though not elegant https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=4095c89a23a2339a1e7afe3813c9acc3
type SharedPtr<T> = Rc<RefCell<T>>;

#[derive(Clone)]
// clone is now inexpensive as we're just referencing the data, not owning it
// we don't want to support all types, we require T to be a number
pub struct Tensor<T: Primitive> {
    pub requires_grad: bool,
    // TODO rename to graph_node
    pub graph: Option<SharedPtr<Node<T>>>,
    pub data: SharedPtr<StorageType<T>>,
    pub grad: SharedPtr<Option<StorageType<f32>>>,
    name: String, // for later use if we want to enforce arg order in ops
}

// TODO a .no_grad() to set all to requires_grad to false and NOT create any graph
impl<T: Primitive, D: Dimension> From<Array<T, D>> for Tensor<T> {
    fn from(arr: Array<T, D>) -> Self {
        // TODO requires_grad false constructor
        Self {
            graph: None,
            data: shared_ptr_new(StorageType::ArrayData(arr.into_dyn())),
            grad: shared_ptr_new(None),
            name: "".to_string(), // TODO switch to id?
            requires_grad: true,
        }
    }
}

impl<T: Primitive> From<StorageType<T>> for Tensor<T> {
    fn from(storage: StorageType<T>) -> Self {
        // here we move and take ownership of the storage
        Tensor {
            graph: None,
            data: shared_ptr_new(storage),
            grad: shared_ptr_new(None),
            name: "".to_string(),
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
            data: shared_ptr_new(StorageType::ArrayData(ArrayD::<T>::zeros(IxDyn(shape)))),
            grad: shared_ptr_new(None),
            name: "".to_string(),
            requires_grad: true,
        }
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            graph: None,
            data: shared_ptr_new(StorageType::ArrayData(ArrayD::<T>::ones(IxDyn(shape)))),
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

    pub fn data(&self) -> Ref<StorageType<T>> {
        self.data.borrow()
    }
    pub fn data_mut(&self) -> RefMut<StorageType<T>> {
        self.data.borrow_mut()
    }

    pub fn grad(&self) -> Ref<Option<StorageType<f32>>> {
        self.grad.borrow()
    }
    pub fn grad_mut(&self) -> RefMut<Option<StorageType<f32>>> {
        self.grad.borrow_mut()
    }

    pub fn move_out_data(&self) -> StorageType<T> {
        // hacky way to move out the value from the RefCell replacing with a dummy value; don't forget to replace it back!
        self.data
            .replace(StorageType::ArrayData(ArrayD::<T>::zeros(IxDyn(&[1]))))
    }

    fn apply(&self, cpu_f: impl Fn(&mut ArrayD<T>)) {
        let mut storage = self.data_mut();
        match &mut *storage {
            StorageType::ArrayData(arr) => cpu_f(arr),
            _ => panic!("Tensors must be on same device"), // TODO return proper result
        }
    }

    pub fn zero_grad(&mut self) {
        // need to get this first or I get a simultaneous borrow and mutation of an object error..
        // let dim = self.data().raw_dim();
        let d = self.data();
        *self.grad.borrow_mut() = Some(StorageType::ArrayData(ArrayD::zeros(d.shape())));
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
        // TODO too many copies
        Tensor::from(self.data().t_clone())
    }
    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }
    pub fn swap_axes(&self, mut ax: i32, mut bx: i32) {
        if ax < 0 {
            ax = self.data().ndim() as i32 + ax;
        }
        if bx < 0 {
            bx = self.data().ndim() as i32 + bx;
        }

        self.apply(|x| x.swap_axes(ax as usize, bx as usize))
    }

    pub fn sum(&self) -> T {
        self.data().sum()
    }

    pub fn as_type<A: Primitive>(&self) -> Tensor<A> {
        // TODO in-place with cast if possible? https://github.com/rust-ndarray/ndarray/issues/493
        // let mut t = Tensor::from(self.data().mapv(|elem| A::from(elem).unwrap()));
        let mut t = Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<T>| x.mapv(|elem| A::from(elem).unwrap()),
            |x| todo!()
        ));
        t.requires_grad = self.requires_grad;
        t
    }

    pub fn fill(&mut self, x: T) {
        self.data_mut().fill(x)
    }
    pub fn is_contiguous(&self) -> bool {
        self.data().is_contiguous()
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
        // NOTE op implementation responsability is a bit of a mess, as I can bypass StorageType;
        // thing is I want a specialized impl of powi for cuda,
        // not a general for loop done with mapv, even if at device level
        // Tensor::from(self.data().mapv(|a| a.powi(exp)))
        Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<f32>| x.mapv(|a| a.powi(exp)),
            |x| todo!()
        ))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.apply(|x| x.mapv_inplace(|a| a.powi(exp)));
        self
    }
}
impl Powi for Tensor<f64> {
    fn powi(&self, exp: i32) -> Self {
        // Tensor::from(self.data().mapv(|a| a.powi(exp)))
        Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<f64>| x.mapv(|a| a.powi(exp)),
            |x| todo!()
        ))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.apply(|x| x.mapv_inplace(|a| a.powi(exp)));
        self
    }
}

// "default" T::pow implementations for other primitives
impl Powi for Tensor<u8> {
    fn powi(&self, exp: i32) -> Self {
        // Tensor::from(self.data().mapv(|a| u8::pow(a, exp as u32)))
        Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<u8>| x.mapv(|a| u8::pow(a, exp as u32)),
            |x| todo!()
        ))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        // self.data_mut().mapv_inplace(|a| u8::pow(a, exp as u32));
        self.apply(|x| x.mapv_inplace(|a| u8::pow(a, exp as u32)));
        self
    }
}

impl Powi for Tensor<i32> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<i32>| x.mapv(|a| i32::pow(a, exp as u32)),
            |x| todo!()
        ))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.apply(|x| x.mapv_inplace(|a| i32::pow(a, exp as u32)));
        self
    }
}
impl Powi for Tensor<i64> {
    fn powi(&self, exp: i32) -> Self {
        Tensor::from(storage_apply!(
            &*self.data(),
            |x: &ArrayD<i64>| x.mapv(|a| i64::pow(a, exp as u32)),
            |x| todo!()
        ))
    }
    fn powi_inplace(self, exp: i32) -> Self {
        self.apply(|x| x.mapv_inplace(|a| i64::pow(a, exp as u32)));
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
        let cpu_dot = |x: &ArrayD<T>, y: &ArrayD<T>| -> Tensor<T> {
            let a = x.view().into_dimensionality::<Ix2>().unwrap();
            let b = y.view().into_dimensionality::<Ix2>().unwrap();
            let res = a.dot(&b);
            Tensor::from(res)
        };
        // let a_ref = self.data();
        // let a = a_ref.view().into_dimensionality::<Ix2>().unwrap();
        // let b_ref = other.data();
        // let b = b_ref.view().into_dimensionality::<Ix2>().unwrap();
        // let res = a.dot(&b);
        // Tensor::from(res)
        storage_apply2!(&*self.data(), &*other.data(), cpu_dot, |x, y| todo!())
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
        self.data().shape()
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

        // move out storage to avoid double borrow
        let storage = self.move_out_data();
        match storage {
            StorageType::ArrayData(arr) => {
                // will panic if the array is *NOT* contiguous
                let reshaped_array = arr.into_shape(shape).unwrap();
                let _ = self.data.replace(StorageType::ArrayData(reshaped_array));
            }
            StorageType::CudaData(_) => todo!(),
        }

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
            Some(ax) => Tensor::from(self.data().mean_axis(Axis(ax))),
            None => Tensor::from(array![self.data().mean()]),
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
        if let StorageType::ArrayData(arr) = tdata {
            let tp: *const ArrayBase<ndarray::OwnedRepr<f64>, Dim<ndarray::IxDynImpl>> =
                arr as *const ArrayD<f64>;
            let t2data = &*t2.data();
            if let StorageType::ArrayData(arr2) = t2data {
                let tp2 = arr2 as *const ArrayD<f64>;
                assert!(tp == tp2);
            }
            assert_eq!(*t.data(), *t2.data());
        }
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
