use ndarray::{Dimension, IntoDimension};
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::autograd::Node;
use crate::tensor::tensor::{ones_like_f32, zeros_like, Tensor};
type SharedPtr<T> = Rc<RefCell<T>>;
// TODO to utils
fn SharedPtrNew<T>(x: T) -> SharedPtr<T> {
    Rc::new(RefCell::new(x))
}

pub trait Operator {
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D>;

    fn backward<T, D>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        grad: Option<SharedPtr<Tensor<f32, D>>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        T: Float + FromPrimitive,
        D: Dimension;

    fn init_backward_grad<T, D>(
        &self,
        vars: &Vec<SharedPtr<Tensor<T, D>>>,
        grad: Option<SharedPtr<Tensor<f32, D>>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        // when grad is already defined, we simply unwrap it
        // otherwise when current Operator is the fist node on which `backward` is called,
        // we create the default "ones" grad tensor for each of the input vars, knowing
        // their size
        match grad {
            Some(g) => vec![g],
            None => {
                let mut grads = Vec::new();
                for input_var in vars {
                    grads.push(SharedPtrNew(ones_like_f32(&input_var.borrow())));
                }
                grads
            }
        }
    }

    // TODO should this function live in autograd? Self isn't even needed!
    fn attach_to_eager_graph<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        op_output: &mut Tensor<T, D>,
        operator: Operators,
    ) {
        // keep references to operator inputs
        let mut vars: Vec<SharedPtr<Tensor<T, D>>> = Vec::new();
        let mut op_parents = Vec::new();
        for x in xs.iter() {
            vars.push(Rc::clone(&x));
            if !x.borrow().graph.is_none() {
                // input is the result of another op, attach current op to it in the graph
                op_parents.push(Rc::clone(x.borrow().graph.as_ref().unwrap()));
            }
            // drop previous references to the graph: `backwards` can only be called from latest output var (e.g. loss)!
            (*x).borrow_mut().graph = None;
        }
        // instatiate new operator node on heap
        let mut op: SharedPtr<Node<T, D>> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            SharedPtrNew(Node::new(operator.into(), vars, Some(op_parents)))
        } else {
            // first node in the graph
            SharedPtrNew(Node::new(operator, vars, None))
        };

        // "attach" output var to graph
        op_output.graph = Some(op);
    }
}

pub struct ReLU;
pub struct Linear;

// TODO solve this variable ordering/naming problem
impl Operator for ReLU {
    fn backward<T, D>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        grad: Option<SharedPtr<Tensor<f32, D>>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        let grads = self.init_backward_grad(&xs, grad);
        {
            let grad = &grads[0];
            let x = xs.get(0).unwrap().borrow();
            let mut g: std::cell::RefMut<'_, Tensor<f32, D>> = grad.borrow_mut();
            // TODO grad[x<=0] = 0
            // for (i, g_i) in g.data.iter_mut().enumerate() {
            for (g_i, x_i) in g.data.iter_mut().zip(x.data.iter()) {
                if *x_i <= T::from_f32(0.).unwrap() {
                    *g_i = 0.0;
                }
            }
        }
        grads
    }
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D> {
        let mut t = zeros_like(&xs[0].borrow());
        // TODO in_place: treat x as output and attach it to current op
        for (tv, xv) in t.data.iter_mut().zip(xs[0].borrow().data.iter()) {
            if *xv > T::from_f32(0.).unwrap() {
                *tv = *xv;
            }
        }
        self.attach_to_eager_graph(xs, &mut t, Operators::ReLU(ReLU));
        t
    }
}

impl Operator for Linear {
    fn backward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        grad: Option<SharedPtr<Tensor<f32, D>>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>> {
        unimplemented!()
    }
    /**
     * x @ W + b
     */
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D> {
        let x: &Tensor<T, D> = &xs[0].borrow();
        // TODO W must be "casted" to a 2D matrix!
        let W = xs[1].borrow();

        unimplemented!();
        // let b = &xs[2].borrow();
        // x.data.dot
        // return x.data.dot(&W) + b;
    }
}

// took a while to figure out a way to deal with dispatching and object safety violated due to generics in traits
// see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=504f29b8003d70964caee607b5655a88
// thanks to https://www.possiblerust.com/pattern/3-things-to-try-when-you-can-t-make-a-trait-object#code-1
pub enum Operators {
    ReLU(ReLU),
    Linear(Linear),
}

impl Into<String> for Operators {
    fn into(self) -> String {
        match (self) {
            Operators::ReLU(_) => String::from("ReLU"),
            Operators::Linear(_) => String::from("Linear"),
            _ => panic!("Unknown operator"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_relu_on_mat() {
        let a = array![[-1., -2.], [3., 4.]];
        let mut x = Tensor::from(a);
        x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;

        let mut xs = vec![Rc::new(RefCell::new(x))];
        let res = ReLU {}.forward(xs);
        for x in &res.data {
            print!("{}\t", x);
        }
        assert!(res.requires_grad);
        assert_eq!(res.data, array![[1., 0.,], [3., 4.]]);
        res.backward();
    }
}
