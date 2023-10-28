use ndarray::Dimension;
use num_traits::{Float, FromPrimitive, Num};
use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::autograd::Node;
use crate::tensor::tensor::{zeros_like, Tensor};
type SharedPtr<T> = Rc<RefCell<T>>;
use ndarray::NdIndex;

pub trait Operator {
    fn forward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
    ) -> Tensor<T, D>;
    // this signature would be needed for setting the value of grad if it was of a generic type
    // fn backward<T, D>(&self, xs: Vec<SharedPtr<Tensor<T, D>>>, grad: SharedPtr<Tensor<f32, D>>)->Vec<SharedPtr<Tensor<T, D>>> ;
    fn backward<T, D>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        grad: SharedPtr<Tensor<f32, D>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        T: Float + FromPrimitive,
        D: Dimension;

    // TODO should this function live in autograd?
    fn attach_to_eager_graph<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        op_output: &mut Tensor<T, D>,
    ) {
        // keep references to operator inputs
        let mut vars: Vec<SharedPtr<Tensor<T, D>>> = Vec::new();
        let mut op_parents = Vec::new();
        for x in xs.iter() {
            vars.push(Rc::clone(&x));
            if !x.borrow().graph.is_none() {
                // input is the result of another op, attach current op to it in the graph
                op_parents.push(Rc::clone(x.borrow().graph.as_ref().unwrap()));
            } else {
                // drop previous references to the graph: `backwards` can only be called from latest output var (e.g. loss)!
                (*x).borrow_mut().graph = None; // TODO isnt this none already?
            }
        }
        // instatiate new operator node on heap
        let mut op: SharedPtr<Node<T, D>> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            Rc::new(RefCell::new(Node::new(
                Operators::ReLU,
                vars,
                Some(op_parents),
            )))
        } else {
            // first node in the graph
            Rc::new(RefCell::new(Node::new(Operators::ReLU, vars, None)))
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
        grad: SharedPtr<Tensor<f32, D>>,
    ) -> Vec<SharedPtr<Tensor<f32, D>>>
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        // having grad as input allows to avoid unnecessary allocations
        {
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
        self.attach_to_eager_graph(xs, &mut t);
        t
    }
}

impl Operator for Linear {
    fn backward<T: Float + FromPrimitive, D: Dimension>(
        &self,
        xs: Vec<SharedPtr<Tensor<T, D>>>,
        grad: SharedPtr<Tensor<f32, D>>,
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
        let x = xs[0].borrow();
        let W = xs[1].borrow();
        let b = &xs[2].borrow();
        unimplemented!();
        // return x.dot(&W) + b;
    }
}

// TODO ReLU(ReLU)
pub enum Operators {
    ReLU,
    Linear,
}
