use ndarray::linalg::Dot;
use ndarray::{Array, Dimension, Ix2, RemoveAxis};
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
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>;

    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>>;

    // TODO should this function live in autograd? Self isn't even needed!
    fn attach_to_eager_graph<T: Float + FromPrimitive>(
        &self,
        mut inputs: Vec<Tensor<T>>,
        op_output: &mut Tensor<T>,
        operator: Operators,
    ) {
        let mut op_parents = Vec::new();
        for x in inputs.iter_mut() {
            if x.graph.is_some() {
                // input is the result of another op, attach current op to it in the graph (through "parents" nodes)
                op_parents.push(Rc::clone(x.graph.as_ref().unwrap()));
            }
            // drop previous references to the graph: `backwards` can only be called from latest output var (e.g. loss)!
            x.graph = None;
        }
        // instantiate new operator node on heap
        let op: SharedPtr<Node<T>> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            shared_ptr_new(Node::new(operator, inputs.clone(), Some(op_parents)))
        } else {
            // first node in the graph
            shared_ptr_new(Node::new(operator, inputs.clone(), None))
        };

        // "attach" output var to graph
        op_output.graph = Some(op);
    }
}

pub struct ReLU;
pub struct Sigmoid;
pub struct Softmax; // TODO
pub struct Mean;
pub struct MeanSquaredError;
pub struct Linear;
pub struct MatMul;

// TODO solve this variable ordering/naming problem
impl Operator for ReLU {
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T> {
        let mut t = zeros_like(&xs[0]);
        // TODO in_place: treat x as output and attach it to current op
        for (tv, xv) in t.data_mut().iter_mut().zip(xs[0].data().iter()) {
            if *xv > T::from_f32(0.).unwrap() {
                *tv = *xv;
            }
        }
        self.attach_to_eager_graph(xs, &mut t, Operators::ReLU(ReLU));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        {
            let x = xs.get(0).unwrap();
            // re-use grad storage
            for (g_i, x_i) in grad.data_mut().iter_mut().zip(x.data().iter()) {
                if *x_i <= 0.0 {
                    *g_i = 0.0;
                }
            }
        }
        vec![grad]
    }
}

impl Operator for Linear {
    /**
     * x @ W + b
     */
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let x: &Tensor<T> = &xs[0];
        let w: &Tensor<T> = &xs[1];
        let b: &Tensor<T> = &xs[2];
        // x.dot creates new tensor, +b: &Tensor adds b to it in-place
        let mut t = x.dot(w) + b;
        self.attach_to_eager_graph(xs, &mut t, Operators::Linear(Linear));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        // if confused->https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/fc_layer

        // b and W will surely need grad (NOTE non learned bias not supported)
        let x: &Tensor<f32> = &xs[0];
        let w: &Tensor<f32> = &xs[1];
        // let b: &Tensor<f32> = &xs[2].borrow();
        let g: &Tensor<f32> = &grad;

        // NOTE in the backward pass, since we need to compute grads as f32 (dot runs with float only),
        // we also need the weights to be f32. In the forward pass (e.g. inference), we can experiment with int only ops
        let dx = g.dot(w); // TODO handle transpose with tensorview
        let dw = g.t_clone().dot(x);
        let db = g.sum_axis(0);
        vec![dx, dw, db]
    }
}

impl Operator for MatMul {
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
        where
            Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let x: &Tensor<T> = &xs[0];
        let w: &Tensor<T> = &xs[1];
        let mut t = x.dot(w);
        self.attach_to_eager_graph(xs, &mut t, Operators::MatMul(MatMul));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let x: &Tensor<f32> = &xs[0];
        let w: &Tensor<f32> = &xs[1];
        let g: &Tensor<f32> = &grad;

        let dx = g.dot(w);
        let dw = g.t_clone().dot(x);
        vec![dx, dw]
    }
}

impl Operator for Sigmoid {
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    {
        let x: &Tensor<T> = &xs[0];
        // TODO too many copies here
        let t: Tensor<f64> = x.as_type();
        t.data_mut().mapv_inplace(|x| 1.0/(1.0 + f64::exp(-x)));
        let mut t: Tensor<T> = t.as_type();
        self.attach_to_eager_graph(xs, &mut t, Operators::MeanSquaredError(MeanSquaredError));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let pred: &Tensor<f32> = &xs[0];
        let target: &Tensor<f32> = &xs[1];
        let g: &Tensor<f32> = &grad;
        // TODO target needs no grad, we could just return one value and let backward_algo zip
        // loop over the shortest array (only works if not required var is last,e.g. zip([x1, x2], [g1]))
        todo!()
        // vec![dx, dw]
    }
}impl Operator for MeanSquaredError {
    /**
        np.mean(np.sum((prediction - target) ** 2, axis=1))
    **/
    fn forward<T: Float + FromPrimitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
        where
            Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let pred: &Tensor<T> = &xs[0];
        let target: &Tensor<T> = &xs[1];
        let mut t = (pred-target).powi_inplace(2).sum_axis(1);
        // TODO macro for this?
        self.attach_to_eager_graph(xs, &mut t, Operators::MeanSquaredError(MeanSquaredError));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let pred: &Tensor<f32> = &xs[0];
        let target: &Tensor<f32> = &xs[1];
        let g: &Tensor<f32> = &grad;
        // TODO target needs no grad, we could just return one value and let backward_algo zip
        // loop over the shortest array (only works if not required var is last,e.g. zip([x1, x2], [g1]))
        todo!()
        // vec![dx, dw]
    }
}

// took a while to figure out a way to deal with dispatching and object safety violated due to generics in traits
// see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=504f29b8003d70964caee607b5655a88
// thanks to https://www.possiblerust.com/pattern/3-things-to-try-when-you-can-t-make-a-trait-object#code-1
pub enum Operators {
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Linear(Linear),
    MatMul(MatMul),
    MeanSquaredError(MeanSquaredError)
}

impl Into<String> for Operators {
    fn into(self) -> String {
        match self {
            Operators::ReLU(_) => String::from("ReLU"),
            Operators::Sigmoid(_) => String::from("Sigmoid"),
            Operators::Linear(_) => String::from("Linear"),
            Operators::MatMul(_) => String::from("MatMul"),
            Operators::MeanSquaredError(_) => String::from("MeanSquaredError"),
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
        let x = Tensor::from(a);
        x.data_mut().view_mut().into_shape(4).unwrap()[0] = 1.0;

        let xs = vec![x];
        let res = ReLU {}.forward(xs);
        for x in res.data().iter() {
            print!("{}\t", x);
        }
        assert!(res.requires_grad);

        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[1., 0.,], [6., 8.]]
        );

        // TODO move to autograd and debug
        res.backward();
    }

    #[test]
    fn test_linear() {
        let w = Tensor::from(array![[1., 1.], [1., 1.]]);
        let b = Tensor::from(array![[1., 1.]]);
        let x = Tensor::from(array![[1., 1.]]);
        let linear = Linear {};
        let xs = vec![x, w, b];
        let res = linear.forward(xs);
        println!("{:?}", res.data);
        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[3., 3.,]]
        );
    }
}
