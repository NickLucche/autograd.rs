use std::ops;
use std::cell::RefCell;
use std::rc::Rc;
use crate::autograd::autograd::{Node, backward_algo};
type SharedPtr<T> = Rc<RefCell<T>>;

// TODO actual tensor lib for data
pub struct Tensor {
    pub requires_grad: bool,
    // TODO rename to graph_node
    // graph: Option<&'graph dyn Operator>
    // I want interior mutability to set only the graph ref, not the rest
    pub graph: Option<SharedPtr<Node>>,
    // graph: Option< Rc<Node> >,
    pub data: Vec<i32>,
    pub grad: Vec<i32>,
    name: String // for later use if we want to enforce arg order in ops
}
impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            requires_grad: false,
            graph: None,
            data: vec![0; 10],
            grad: vec![0; 10],
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
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(mut self, rhs: Tensor) -> Tensor {
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
        self
    }
}

// TODO other file for ops
pub fn ones() -> Tensor {
    Tensor::new()
}