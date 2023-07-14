use crate::operators::operators::{Operator, Operators, ReLU};
use crate::tensor::tensor::{ones, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
// NOTE not thread-safe!
type SharedPtr<T> = Rc<RefCell<T>>;

pub struct Node {
    pub name: Operators,
    pub variables: Vec<SharedPtr<Tensor>>,
    pub parents: Option<Vec<SharedPtr<Node>>>,
}

impl Node {
    pub fn new(
        name: Operators, // TODO either name mapping lazy op creation or impl Operator trait
        variables: Vec<SharedPtr<Tensor>>,
        parents: Option<Vec<SharedPtr<Node>>>,
    ) -> Self {
        Node {
            name,
            variables,
            parents,
        }
    }
    fn accumulate_grad(&self, grad: Tensor) -> Tensor {
        !unimplemented!()
    }
    fn get_grad(&self) -> Tensor {
        !unimplemented!()
    }
}

// fn attach_to_graph(graph_node: &Rc<Node>, op: &mut Rc<Node>) {
//     match &mut op.parents {
//         None => op.parents = Some(vec![Rc::clone(graph_node)]),
//         Some(nodes)=>nodes.push(Rc::clone(graph_node))
//     }
// }

pub fn backward_algo(node: SharedPtr<Node>, prev_grad: Option<SharedPtr<Tensor>>) {
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
        // var.borrow_mut().grad += grads[i];
        if var.borrow().requires_grad {
            let acc = &mut var.borrow_mut().grad;
            for j in 0..acc.len() {
                acc[j] += grads[i].borrow().data[j];
            }
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
    fn test_foo() {}
}
