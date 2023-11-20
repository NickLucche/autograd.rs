use ndarray::linalg::Dot;
use ndarray::{Array, Dimension, RemoveAxis};
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::autograd::Node;
use crate::tensor::tensor::{ones_like_f32, zeros_like, Tensor};
type SharedPtr<T> = Rc<RefCell<T>>;

// TODO to utils
pub fn shared_ptr_new<T>(x: T) -> SharedPtr<T> {
    Rc::new(RefCell::new(x))
}

pub trait Operator {
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D>
    where
        Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>;

    fn backward<D>(
        &self,
        xs: Vec<SharedPtr<Tensor<f32, D>>>,
        grad: SharedPtr<Tensor<f32, D>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        D: Dimension,
        Array<f32, D>: Dot<Array<f32, D>, Output = Array<f32, D>>;

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
        let op: SharedPtr<Node<T, D>> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            shared_ptr_new(Node::new(operator.into(), vars, Some(op_parents)))
        } else {
            // first node in the graph
            shared_ptr_new(Node::new(operator, vars, None))
        };

        // "attach" output var to graph
        op_output.graph = Some(op);
    }
}

pub struct ReLU;
pub struct Linear;

// TODO solve this variable ordering/naming problem
impl Operator for ReLU {
    fn backward<D>(
        &self,
        xs: Vec<SharedPtr<Tensor<f32, D>>>,
        grad: SharedPtr<Tensor<f32, D>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        D: Dimension,
        Array<f32, D>: Dot<Array<f32, D>, Output = Array<f32, D>>
    {
        {
            let x = xs.get(0).unwrap().borrow();
            let mut g: std::cell::RefMut<'_, Tensor<f32, D>> = grad.borrow_mut();
            // TODO grad[x<=0] = 0
            // for (i, g_i) in g.data.iter_mut().enumerate() {
            for (g_i, x_i) in g.data.iter_mut().zip(x.data.iter()) {
                if *x_i <= 0.0 {
                    *g_i = 0.0;
                }
            }
        }
        vec![grad]
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
    fn backward<D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<f32, D>>>,
        grad: SharedPtr<Tensor<f32, D>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        Array<f32, D>: Dot<Array<f32, D>, Output = Array<f32, D>>
    {
        // if confused->https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/fc_layer

        // b and W will surely need grad (NOTE non learned bias not supported)
        let x: &Tensor<f32, D> = &xs[0].borrow();
        let w: &Tensor<f32, D> = &xs[1].borrow();
        // let b: &Tensor<f32, D> = &xs[2].borrow();
        let g: &Tensor<f32, D> = &grad.borrow();

        unimplemented!()
        // NOTE in the backward pass, since we need to compute grads as f32 (dot runs with float only),
        // we also need the weights to be f32. In the forward pass (e.g. inference), we can experiment with int only ops
        // let dx = g.dot(w); // TODO handle traspose with tensorview
        // let dw = x.dot(g);
        // let db = g.sum_axis(0);

        // vec![dx, dw, db].into_iter().map(|x| shared_ptr_new(x)).collect()
    }
    /**
     * x @ W + b
     */
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D>
    where
        Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
    {
        let x: &Tensor<T, D> = &xs[0].borrow();
        let w = xs[1].borrow();
        let b: &Tensor<T, D> = &xs[2].borrow();
        // x.dot creates new tensor, +b: &Tensor adds b to it in-place
        x.dot(&w) + b
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
        match self {
            Operators::ReLU(_) => String::from("ReLU"),
            Operators::Linear(_) => String::from("Linear"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_relu_on_mat() {
        let mut a = array![[-1., -2.], [3., 4.]];
        let b = array![[-1., -2.], [3., 4.]];
        a += &b;
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

    #[test]
    fn test_linear() {
        let w = Tensor::from(array![[1., 1.], [1., 1.]]);
        let b = Tensor::from(array![[1., 1.]]);
        let x = Tensor::from(array![[1., 1.]]);
        let linear = Linear {};
        let xs = vec![
            Rc::new(RefCell::new(x)),
            Rc::new(RefCell::new(w)),
            Rc::new(RefCell::new(b)),
        ];
        let res = linear.forward(xs);
        println!("{:?}", res.data);
    }
}
