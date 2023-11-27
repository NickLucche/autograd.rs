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
    pub variables: Vec<SharedPtr<Tensor<T>>>,
    pub parents: Option<Vec<SharedPtr<Node<T>>>>,
}

impl<T> Node<T>
where
    T: Float + FromPrimitive,
{
    pub fn new(
        operator: Operators,
        variables: Vec<SharedPtr<Tensor<T>>>,
        parents: Option<Vec<SharedPtr<Node<T>>>>,
    ) -> Self {
        Node {
            operator,
            variables,
            parents,
        }
    }
}

pub fn backward_algo(node: SharedPtr<Node<f32>>, prev_grad: SharedPtr<Tensor<f32>>)
where
    Array<f32, Ix2>: Dot<Array<f32, Ix2>, Output = Array<f32, Ix2>>,
{
    // 1. compute gradient(s) of current operator wrt its input(s)
    let op = &node.borrow().operator;
    // TODO avoid computing grad altogheter if var does not require it
    let op_inputs = node.borrow().variables.to_vec(); // TODO this does a copy!
                                                      // manual dispatch with lazy init of grad
    let grads = match op {
        Operators::ReLU(op) => op.backward(op_inputs, prev_grad),
        Operators::Linear(op) => op.backward(op_inputs, prev_grad),
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
        backward_algo(Rc::clone(&parent), g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::operators::ReLU;
    use ndarray::prelude::*;

    #[test]
    fn test_simple_graph() {
        let a = array![[0., 1.], [2., 3.]];
        let x = Tensor::from(a);
        let x2 = x.clone();
        // x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;
        let xs = vec![Rc::new(RefCell::new(x))];
        let res = ReLU {}.forward(xs);
        for x in &res.data {
            print!("{}\t", x);
        }
        assert_eq!(res.data, x2.data);
        res.backward();
    }
}
