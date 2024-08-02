use crate::autograd::autograd::{backward_algo, Node};
use crate::operators::operators::shared_ptr_new;
use ndarray::linalg::Dot;
use ndarray::{array, Array, ArrayD, ArrayView, ArrayViewMut, Axis, Dimension, Ix2, IxDyn};
use ndarray::iter::{IterMut, Iter};
use std::cell::{Ref, RefCell, RefMut};
use std::convert::From;
use std::ops::AddAssign;
use std::rc::Rc;

extern crate num_traits;

use super::init::{kaiming_uniform, uniform};
use super::storage;
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

#[derive(Clone, Debug)]
pub struct CudaData<T> {
    pub ptr: *mut T,
}

impl<T> Drop for CudaData<T> {
    fn drop(&mut self) {
        // deallocate_cuda_memory(self.ptr);  // Replace with your actual deallocation function
    }
}

#[macro_export]
macro_rules! storage_apply {
    ($value:expr, $func_array:expr, $func_cuda:expr) => {
        match $value {
            StorageType::ArrayData(arr) => $func_array(arr),
            StorageType::CudaData(arr) => $func_cuda(arr),
        }
    };
}

#[macro_export]
macro_rules! storage_apply2 {
    ($value1:expr, $value2:expr, $func_array:expr, $func_cuda:expr) => {
        match ($value1, $value2) {
            (StorageType::ArrayData(arr_a), StorageType::ArrayData(arr_b)) => {
                $func_array(arr_a, arr_b)
            }
            (StorageType::CudaData(cuda_a), StorageType::CudaData(cuda_b)) => {
                $func_cuda(cuda_a, cuda_b)
            }
            _ => panic!("Tensors must be on same device"),
        }
    };
}
// TODO move into own file
#[derive(Clone, Debug)]
pub enum StorageType<T> {
    ArrayData(Array<T, IxDyn>), // CPU
    CudaData(CudaData<T>),      // CUDA
}

impl<T: Primitive> StorageType<T> {
    pub fn t(&self) -> StorageType<T> {
        todo!()
    }
    pub fn ndim(&self) -> usize {
        storage_apply!(&self, |x: &ArrayD<T>| x.ndim(), |x: &CudaData<T>| todo!())
    }
    pub fn shape(&self) -> Vec<usize> {
        storage_apply!(&self, |x: &ArrayD<T>| x.shape().to_vec(), |x: &CudaData<
            T,
        >| todo!())
    }
    pub fn raw_dim(&self) -> Vec<usize> {
        // on raw_dim/shape difference https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.shape
        // all our arrays have dyn shape anyway
        self.shape()
    }
    pub fn is_contiguous(&self) -> bool {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.is_standard_layout(),
            |x: &CudaData<T>| true
        )
    }
    pub fn is_empty(&self) -> bool {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| self.len()==0,
            |x: &CudaData<T>| self.len()==0
        )
    }
    pub fn fill(&mut self, el: T) {
        storage_apply!(self, |x: &mut ArrayD<T>| x.fill(el), |x: &mut CudaData<
            T,
        >| todo!())
    }
    pub fn len(&self) -> usize {
        storage_apply!(&self, |x: &ArrayD<T>| x.len(), |x: &CudaData<T>| todo!())
    }

    // ops
    pub fn sum(&self) -> T {
        storage_apply!(&self, |x: &ArrayD<T>| x.sum(), |x: &CudaData<T>| todo!())
    }
    pub fn sum_axis(&self, a: Axis) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.sum_axis(a),
            |x: &CudaData<T>| todo!()
        )
    }
    pub fn mean(&self) -> T {
        storage_apply!(&self, |x: &ArrayD<T>| x.mean().unwrap(), |x: &CudaData<
            T,
        >| todo!())
    }
    pub fn mean_axis(&self, a: Axis) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.mean_axis(a).unwrap(),
            |x: &CudaData<T>| todo!()
        )
    }

    pub fn broadcast(&self, a: IxDyn) -> ArrayD<T> {
        storage_apply!(
            &self,
            |x: &ArrayD<T>| x.broadcast(a).unwrap().to_owned(),
            |x: &CudaData<T>| todo!()
        )
    }

    // TODO dont want to implement these for cuda tbh, ndarray dispatch should be handled by caller, hence they just return ndarrays
    pub fn mapv(&self, f: impl Fn(T) -> T) -> ArrayD<T> {
        storage_apply!(&self, |x: &ArrayD<T>| x.mapv(f), |x: &CudaData<T>| todo!())
    }

    pub fn mapv_inplace(&mut self, f: impl Fn(T) -> T) {
        storage_apply!(self, |x: &mut ArrayD<T>| x.mapv_inplace(f), |x: &mut CudaData<T>| todo!())
    }

    pub fn map(&self, f: impl Fn(&T) -> T) -> ArrayD<T> {
        storage_apply!(&self, |x: &ArrayD<T>| x.map(f), |x: &CudaData<T>| todo!())
    }

    pub fn view(&self) -> ArrayView<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.view()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn view_mut(&mut self) -> ArrayViewMut<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.view_mut()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn iter(&self) -> Iter<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.iter()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T, IxDyn> {
        if let StorageType::ArrayData(arr) = self {
            arr.iter_mut()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }

    pub fn into_raw_vec(&self) -> Vec<T> {
        if let StorageType::ArrayData(arr) = self {
            arr.clone().into_raw_vec()
        } else {
            panic!("Not Implemented for CudaData")
        }
    }
    
}


#[derive(Clone)]
pub enum Device {
    CPU,
    CUDA,
}

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
        let cpu_foo = |x: &ArrayD<T>| -> Tensor<T> {
            Tensor {
                graph: None,
                data: shared_ptr_new(StorageType::ArrayData(x.to_owned())), // TODO no copy?
                grad: shared_ptr_new(None),
                name: "".to_string(),
                requires_grad: true,
            }
        };
        let cuda_foo = |x: &CudaData<T>| -> Tensor<T> { todo!() };
        storage_apply!(&storage, cpu_foo, cuda_foo)
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
        Tensor::from(self.data().t().to_owned())
    }
    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }
    pub fn swap_axes(&self, mut ax: i32, mut bx: i32) {
        let t = self.data_mut();
        if ax < 0 {
            ax = t.ndim() as i32 + ax;
        }
        if bx < 0 {
            bx = t.ndim() as i32 + bx;
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

        // can't move with into_inner/take..
        let d = self.data();
        match &*d {
            StorageType::ArrayData(arr) => {
                let array = self
                    .data
                    .replace(StorageType::ArrayData(ArrayD::<T>::zeros(IxDyn(&[1]))));
                // NOTE will panic if the array is *NOT* contiguous
                if let StorageType::ArrayData(arr) = array {
                    let reshaped_array = arr.into_shape(shape).unwrap();
                    let _ = self.data.replace(StorageType::ArrayData(reshaped_array));
                }
            }
            StorageType::CudaData(_) => todo!(),
        }
        // let array = self.data.replace(ArrayD::<T>::zeros(IxDyn(&[1])));
        // let reshaped_array = array.into_shape(shape).unwrap();
        // let _ = self.data.replace(reshaped_array);

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
            let tp: *const ArrayBase<ndarray::OwnedRepr<f64>, Dim<ndarray::IxDynImpl>> = arr as *const ArrayD<f64>;
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
