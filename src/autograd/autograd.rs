use crate::operators::operators::{Operator, Operators};
use crate::tensor::Primitive;
use crate::tensor::tensor::{Powi, Tensor};
use ndarray::linalg::Dot;
use ndarray::{Array, Dimension, Ix2};
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::rc::Rc;
// NOTE not thread-safe!
type SharedPtr<T> = Rc<RefCell<T>>;

pub struct Node<T: Primitive> {
    pub operator: Operators,
    pub variables: Vec<Tensor<T>>,
    // keep reference to the op which generated each input var, if any!
    // (e.g. "weights" have no parent node in graph)
    // this allows to disambiguate ordering for mixed op results and plain var
    // inputs e.g MatMul(x, Op(__))!=MatMul(Op(__), x)
    pub parents: Vec<Option<SharedPtr<Node<T>>>>,
}

impl<T> Node<T>
where
    T: Primitive,
{
    pub fn new(
        operator: Operators,
        variables: Vec<Tensor<T>>,
        parents: Vec<Option<SharedPtr<Node<T>>>>,
    ) -> Self {
        Node {
            operator,
            variables,
            parents,
        }
    }

    pub fn is_root_node(&self)->bool {
        // root nodes in graph have no parents
        for parent in &self.parents {
            if parent.is_some() {
                return false;
            }
        }
        true
    }
}

pub fn backward_algo(node: SharedPtr<Node<f32>>, prev_grad: Tensor<f32>)
where
    Array<f32, Ix2>: Dot<Array<f32, Ix2>, Output = Array<f32, Ix2>>, Tensor<f32>: Powi
{
    let mut node = node.borrow_mut();
    // 1. compute gradient(s) of current operator wrt its input(s)
    let op = &node.operator;
    // TODO avoid computing grad altogether if var does not require it: resp would lie with operators
    let op_inputs = node.variables.to_vec(); // copy is fine with tensors

    // manual dispatch with lazy init of grad
    let grads = match op {
        Operators::ReLU(op) => op.backward(op_inputs, prev_grad),
        Operators::Sigmoid(op) => op.backward(op_inputs, prev_grad),
        Operators::MatMul(op) => op.backward(op_inputs, prev_grad),
        Operators::Linear(op) => op.backward(op_inputs, prev_grad),
        Operators::MeanSquaredError(op) => op.backward(op_inputs, prev_grad),
        Operators::Mean(op) => op.backward(op_inputs, prev_grad),
        Operators::Identity(op) => op.backward(op_inputs, prev_grad),
        Operators::Conv2D(op) => op.backward(op_inputs, prev_grad),
    };
    // TODO this assumes that the node computes a gradient for each input! This is not true for e.g losses..
    // can be resolved returning an optional on grad, and do a check if requires_grad within backward to solve todos above
    // 2. accumulate gradient on input vars
    for ( var, grad) in node.variables.iter_mut().zip(grads.iter()) {
        // println!("GRAD SHAPE {:?}, VAR SHAPE {:?}", grad.shape(), var.shape());
        if var.requires_grad {
            // lazy init of x grad when accumulating
            var.accumulate_grad_from_grad_tensor(grad);
        }
    }
    // 3. recurse on parent nodes
    for (i, parent) in node.parents.iter().enumerate() {
        if let Some(p) = parent {
            backward_algo(Rc::clone(p), grads[i].clone()); // safe to clone tensors
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::operators::{Linear, ReLU};
    use ndarray::prelude::*;

    #[test]
    fn test_single_node_graph() {
        let a = array![[0., -1.], [2., 3.]];
        let x = Tensor::from(a);
        // x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;
        let xs = vec![x];
        let res = ReLU {}.forward(xs.clone()); // TODO impl copy?
        for x in res.data().iter() {
            print!("{}\t", x);
        }
        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[0., 0.,], [2., 3.]]
        );
        res.backward();
        let g = &xs[0];
        assert_eq!(
            g.grad()
                .as_ref()
                .unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            array![[0., 0.,], [1., 1.]]
        );
    }

    #[test]
    fn test_simple_graph() {
        // 1x2
        let x = Tensor::from(array![[0., 1.]]);
        let x_copy = x.clone();
        let xs = vec![x];
        let res = ReLU {}.forward(xs);
        // 2x2
        let w = Tensor::from(array![[1., 1.], [1., 1.]]);
        // 1x2
        let b = Tensor::from(array![[1., 1.]]);
        let xs = vec![res, w, b];

        let res = Linear {}.forward(xs.clone());
        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[2., 2.,]]
        );
        res.backward();
        assert_eq!(
            x_copy
                .grad()
                .as_ref()
                .unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            array![[0., 2.]]
        );
        assert_eq!(
            xs[1]
                .grad()
                .as_ref()
                .unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            array![[0., 0.], [1., 1.]]
        );
        // TODO should we reshape to 2dim?
        assert_eq!(
            xs[2]
                .grad()
                .as_ref()
                .unwrap()
                .view()
                .into_dimensionality::<Ix1>()
                .unwrap(),
            array![1., 1.]
        );
    }
}
