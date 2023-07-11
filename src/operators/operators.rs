use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::tensor::Tensor;
use crate::autograd::autograd::Node;
type SharedPtr<T> = Rc<RefCell<T>>;

pub trait Operator {
    fn forward(&self, xs: Vec<SharedPtr<Tensor>>) -> Tensor;
    fn backward(&self, xs: Vec<SharedPtr<Tensor>>, grad: SharedPtr<Tensor>)->Vec<SharedPtr<Tensor>>;

    // TODO should this function live in autograd?
    fn attach_to_eager_graph(&self, xs: Vec<SharedPtr<Tensor>>, op_output: &mut Tensor) {
        // keep references to operator inputs
        let mut vars: Vec<SharedPtr<Tensor>> = Vec::new();
        let mut op_parents = Vec::new();
        for x in xs.iter() {
            vars.push(Rc::clone(&x));
            if !x.borrow().graph.is_none() {
                // input is the result of another op, attach current op to it in the graph
                op_parents.push(Rc::clone(x.borrow().graph.as_ref().unwrap()));
            } else {
                // drop previous references to the graph: `backwards` can only be called from latest output var (e.g. loss)!
                (*x).borrow_mut().graph = None;
            }
        }
        // instatiate new operator node on heap
        let mut op: SharedPtr<Node> = if op_parents.len() > 0 {
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
    fn backward(&self, xs: Vec<SharedPtr<Tensor>>, grad: SharedPtr<Tensor>)->Vec<SharedPtr<Tensor>> {
        // having grad as input allows to avoid unnecessary allocations
        let x = xs[0].borrow();
        let mut g: std::cell::RefMut<'_, Tensor> = grad.borrow_mut();
        for i in 0..x.data.len() {
            if x.data[i] <= 0 {
                g.data[i] = 0;
            }
        }
        vec![Rc::clone(&grad)]
    }
    fn forward(&self, xs: Vec<SharedPtr<Tensor>>) -> Tensor {
        let mut t = Tensor::new();
        // TODO in_place: treat x as output and attach it to current op
        for (i, val) in xs[0].borrow_mut().data.iter().enumerate() {
            if *val > 0 {
                t.data[i] = *val;
            }
        }
        self.attach_to_eager_graph(xs, &mut t);
        t
    }
}

impl Operator for Linear {
    fn backward(&self, xs: Vec<SharedPtr<Tensor>>, grad: SharedPtr<Tensor>)->Vec<SharedPtr<Tensor>> {
        unimplemented!()
    }
    /**
     * x @ W + b
     */
    fn forward(&self, xs: Vec<SharedPtr<Tensor>>) -> Tensor {
        let x = xs[0].borrow();
        let W = xs[1].borrow();
        let b = &xs[2].borrow();

        return x.dot(&W) + b;
    }
}

// TODO ReLU(ReLU)
pub enum Operators {
    ReLU,
    Linear
}