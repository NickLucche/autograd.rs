use crate::operators::operators::{Operator, Operators};
use crate::tensor::tensor::Tensor;
use ndarray::linalg::Dot;
use ndarray::{Array, Dimension, Ix2};
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::rc::Rc;
// NOTE not thread-safe!
type SharedPtr<T> = Rc<RefCell<T>>;

pub struct Node<T: Float + FromPrimitive> {
    pub operator: Operators,
    pub variables: Vec<Tensor<T>>,
    pub parents: Option<Vec<SharedPtr<Node<T>>>>,
}

impl<T> Node<T>
where
    T: Float + FromPrimitive,
{
    pub fn new(
        operator: Operators,
        variables: Vec<Tensor<T>>,
        parents: Option<Vec<SharedPtr<Node<T>>>>,
    ) -> Self {
        Node {
            operator,
            variables,
            parents,
        }
    }
}

pub fn backward_algo(node: SharedPtr<Node<f32>>, prev_grad: Tensor<f32>)
where
    Array<f32, Ix2>: Dot<Array<f32, Ix2>, Output = Array<f32, Ix2>>,
{
    let mut node = node.borrow_mut();
    // 1. compute gradient(s) of current operator wrt its input(s)
    let op = &node.operator;
    // TODO avoid computing grad altogether if var does not require it
    let op_inputs = node.variables.to_vec(); // copy is fine with tensors
                                             // manual dispatch with lazy init of grad
    let grads = match op {
        Operators::ReLU(op) => op.backward(op_inputs, prev_grad),
        Operators::Linear(op) => op.backward(op_inputs, prev_grad),
    };
    // 2. accumulate gradient on input vars
    for (i, var) in node.variables.iter_mut().enumerate() {
        // TODO should I check for `requires_grad` inside accumulate and silently do nothing?
        if var.requires_grad {
            // lazy init of x grad when accumulating
            var.accumulate_grad_from_grad_tensor(&grads[i]);
        }
    }
    // 3. recurse on parent nodes
    for (i, parent) in node.parents.as_ref().unwrap_or(&vec![]).iter().enumerate() {
        backward_algo(Rc::clone(parent), grads[i].clone()); // safe to clone tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::operators::{ReLU, Linear};
    use ndarray::prelude::*;

    #[test]
    fn test_single_node_graph() {
        let a = array![[0., -1.], [2., 3.]];
        let x = Tensor::from(a);
        // x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;
        let xs = vec![x];
        let res = ReLU{}.forward(xs.clone()); // TODO impl copy?
        for x in res.data().iter() {
            print!("{}\t", x);
        }
        assert_eq!(res.data().view().into_dimensionality::<Ix2>().unwrap(), array![[0., 0.,], [2., 3.]]);
        res.backward();
        let g = &xs[0];
        assert_eq!(g.grad().as_ref().unwrap().view().into_dimensionality::<Ix2>().unwrap(), array![[0., 0.,], [1., 1.]]);
    }

    #[test]
    fn test_simple_graph() {
        let x = Tensor::from(array![[0., 1.]]);
        let x_copy = x.clone();
        let xs = vec![x];
        let res = ReLU{}.forward(xs);

        let w = Tensor::from(array![[1., 1.], [1., 1.]]);
        let b = Tensor::from(array![[1., 1.]]);
        let xs = vec![res, w, b];

        let res = Linear{}.forward(xs.clone());
        assert_eq!(res.data().view().into_dimensionality::<Ix2>().unwrap(), array![[2., 2.,]]);
        res.backward();
        assert_eq!(x_copy.grad().as_ref().unwrap().view().into_dimensionality::<Ix2>().unwrap(), array![[0., 2.]]);
        assert_eq!(xs[1].grad().as_ref().unwrap().view().into_dimensionality::<Ix2>().unwrap(), array![[0., 1.], [0., 1.]]);
        // TODO should we reshape to 2dim?
        assert_eq!(xs[2].grad().as_ref().unwrap().view().into_dimensionality::<Ix1>().unwrap(), array![1., 1.]);
    }
}
