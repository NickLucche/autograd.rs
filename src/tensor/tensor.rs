use std::ops;
use std::cell::RefCell;
use std::rc::Rc;
use std::convert::From;
use ndarray::{Array, ArrayBase, array, Dimension};
use crate::autograd::autograd::{Node, backward_algo};

type SharedPtr<T> = Rc<RefCell<T>>;
// ndarray extension inspired by https://github.com/raskr/rust-autograd

pub struct Tensor<T, D: Dimension> {
    pub requires_grad: bool,
    // TODO rename to graph_node
    pub graph: Option<SharedPtr<Node<T, D>>>,
    // graph: Option< Rc<Node> >,
    pub data: Array<T, D>,
    pub grad: Array<T, D>,
    name: String // for later use if we want to enforce arg order in ops
}
impl<T, D: Dimension> From<Array<T, D> > for Tensor<T, D> {
    fn from(arr: Array<T, D>) -> Self {
        Self {
            graph: None,
            data: arr,
            grad: Array::<f32, D>::zeros(arr.shape()),
            name: "".to_string(),
            requires_grad: false
        }
    }
}
impl<D> Tensor<f32, D> {
    fn new2() -> Self {
        Self {
            graph: None,
            data: Array::<f32, D>::ones((2, 4)),
            grad: Array::<f32, D>::ones((2, 4)),
            name: "".to_string(),
            requires_grad: false
        }
    }
}
impl<T, D> Tensor<T, D> {
    pub fn new() -> Tensor<T, D> {
        Tensor {
            requires_grad: false,
            graph: None,
            data: array![[]],
            grad: array![[]],
            name: "".to_string()
        }
    }

    pub fn backward(&self) {
        if !self.requires_grad {
            unimplemented!();
        }
        match &self.graph {
            Some(g)=>backward_algo(Rc::clone(g), None),
            _ => {}
        }
    }

    pub fn dot(&self, other: &Tensor<T, D>)->Tensor<T, D> {
        unimplemented!()
    }
}


// TODO other file for ops
pub fn ones<T, D>() -> Tensor<T, D> {
    Tensor::new()
}

impl<T, D> From<Array<T, D>> for Tensor<T, D> {
    fn from(value: Array<T, D>) -> Self {
        Tensor { requires_grad: false, graph: None, data: value, grad: array![[]], name: "".to_string() }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_froms() {

    }
}