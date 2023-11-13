use ndarray::Dimension;
use num_traits::{FromPrimitive, Float};

use crate::operators::operators::{Operator, Operators, ReLU};
use crate::tensor::tensor::{ones, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
// NOTE not thread-safe!
type SharedPtr<T> = Rc<RefCell<T>>;

pub struct Node<T: Float+FromPrimitive, D: Dimension> {
    pub operator: Operators,
    pub variables: Vec<SharedPtr<Tensor<T, D>>>,
    pub parents: Option<Vec<SharedPtr<Node<T, D>>>>,
}

impl<T, D> Node<T, D> where T: Float+FromPrimitive, D: Dimension {
    pub fn new(
        operator: Operators,
        variables: Vec<SharedPtr<Tensor<T, D>>>,
        parents: Option<Vec<SharedPtr<Node<T, D>>>>,
    ) -> Self {
        Node {
            operator,
            variables,
            parents,
        }
    }
    fn accumulate_grad(&self, grad: Tensor<T, D>) -> Tensor<T, D> {
        !unimplemented!()
    }
    fn get_grad(&self) -> Tensor<T, D> {
        !unimplemented!()
    }
}


pub fn backward_algo<T: Float+FromPrimitive, D: Dimension>(node: SharedPtr<Node<T, D>>, prev_grad: Option<SharedPtr<Tensor<f32, D>>>) {
    let prev_grad = prev_grad.unwrap_or(Rc::new(RefCell::new(ones())));
    // 1. compute gradient(s) of current operator wrt its input(s)
    // lazy init
    let op = &node.borrow().operator;
    // TODO avoid computing grad altogheter if var does not require it
    let op_inputs = node.borrow().variables.to_vec(); // TODO this does a copy!
    // manual dispatch
    let grads = match op {
        Operators::ReLU(op)=>op.backward(op_inputs, prev_grad),
        Operators::Linear(op)=>op.backward(op_inputs, prev_grad)
    };
    // 2. accumulate gradient on input vars
    for (i, var) in node.borrow().variables.iter().enumerate() {
        // TODO should I check for `requires_grad` inside accumulate and silently do nothing? 
        if var.borrow().requires_grad {
            let x = &mut var.borrow_mut();
            // lazy init of x grad when accumulating
            x.accumulate_grad(&grads[i].borrow());
        }
    }
    // 3. recurse on parent nodes
    for (i, parent) in node
        .borrow()
        .parents
        .as_ref()
        .unwrap_or(&vec![])
        .iter()
        .enumerate()
    {
        let g = Rc::clone(&grads[i]);
        backward_algo(Rc::clone(&parent), Some(g));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_simple_graph() {
        let a = array![[0.,1.], [2., 3.]];
        let mut x = Tensor::from(a);
        let x2 = x.clone();
        // x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;
        let mut xs = vec![Rc::new(RefCell::new(x))];
        let res = ReLU{}.forward(xs);
        for x in &res.data {
            print!("{}\t", x);
        }
        assert_eq!(res.data, x2.data);
    }
}
