use ndarray::Dimension;
use num_traits::{FromPrimitive, Float};

use crate::operators::operators::{Operator, Operators, ReLU};
use crate::tensor::tensor::{ones, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
// NOTE not thread-safe!
type SharedPtr<T> = Rc<RefCell<T>>;

pub struct Node<T: Float+FromPrimitive, D: Dimension> {
    pub name: Operators,
    pub variables: Vec<SharedPtr<Tensor<T, D>>>,
    pub parents: Option<Vec<SharedPtr<Node<T, D>>>>,
}

impl<T, D> Node<T, D> where T: Float+FromPrimitive, D: Dimension {
    pub fn new(
        name: Operators, // TODO either name mapping lazy op creation or impl Operator trait
        variables: Vec<SharedPtr<Tensor<T, D>>>,
        parents: Option<Vec<SharedPtr<Node<T, D>>>>,
    ) -> Self {
        Node {
            name,
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

// fn attach_to_graph<T, D>(graph_node: &Rc<Node>, op: &mut Rc<Node>) {
//     match &mut op.parents {
//         None => op.parents = Some(vec![Rc::clone(graph_node)]),
//         Some(nodes)=>nodes.push(Rc::clone(graph_node))
//     }
// }

pub fn backward_algo<T: Float+FromPrimitive, D: Dimension>(node: SharedPtr<Node<T, D>>, prev_grad: Option<SharedPtr<Tensor<f32, D>>>) {
    let prev_grad = prev_grad.unwrap_or(Rc::new(RefCell::new(ones())));
    // 1. compute gradient(s) of current operator wrt its input(s)
    // FIXME from trait node.name->operator
    // lazy init
    let op = ReLU {};
    // TODO avoid computing grad altogheter if var does not require it
    let op_inputs = node.borrow().variables.to_vec(); // TODO this does a copy!
    let grads = op.backward(op_inputs, prev_grad);
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

// TODO unit tests here
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {}
}
