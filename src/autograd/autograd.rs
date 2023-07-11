use std::cell::RefCell;
use std::ops;
use std::rc::Rc;

// TODO actual tensor lib for data
// TODO this thing might as well go on heap, since data itself does and use Box<Tensor> around
pub struct Tensor {
    requires_grad: bool,
    // TODO rename to graph_node
    // graph: Option<&'graph dyn Operator>
    // I want interior mutability to set only the graph ref, not the rest
    graph: Option<Rc<RefCell<Node>>>,
    // graph: Option< Rc<Node> >,
    pub data: Vec<i32>,
    grad: Vec<i32>,
}
impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            requires_grad: false,
            graph: None,
            data: vec![0; 10],
            grad: vec![0; 10],
        }
    }

    pub fn backward(&self) {
        if !self.requires_grad {
            unimplemented!();
        }
        match &self.graph {
            Some(g)=>backward_algo(Rc::clone(g), None),
            _ => {}
        }
    }
}
impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(mut self, rhs: Tensor) -> Tensor {
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
        self
    }
}

// TODO alternative->composition: have a OperatorImpl
// type that defines functions with grad and parents and

// struct OperatorImpl {
//     grad: Tensor,
//     parents: Option<Vec<Tensor>>
// }

// TODO we should say 'inputs > 'node, outlives it
struct Node {
    pub name: OperatorNodes,
    // TODO an operator doesn't need a grad cache it can just forward it, the variable does
    pub grad: Tensor,
    // TODO swap with Tensor
    pub variables: Vec<Rc<RefCell<Tensor>>>,
    pub parents: Option<Vec<Rc<RefCell<Node>>>>,
}

impl Node {
    fn new(
        name: OperatorNodes,
        variables: Vec<Rc<RefCell<Tensor>>>,
        parents: Option<Vec<Rc<RefCell<Node>>>>,
    ) -> Self {
        Node {
            name,
            grad: Tensor::new(),
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

pub struct ReLU;
// TODO backward algo
fn ones() -> Tensor {
    Tensor::new()
}
fn foo(xs: Vec<Rc<RefCell<Tensor>>>, g: Rc<RefCell<Tensor>>)->Vec<Rc<RefCell<Tensor>>>{
    g.borrow_mut().data[0] = 1;
    vec![ Rc::new(RefCell::new(Tensor::new())) ]
}
fn backward_algo(node: Rc<RefCell<Node>>, prev_grad: Option<Rc<RefCell<Tensor>>>) {
    let prev_grad = prev_grad.unwrap_or(Rc::new(RefCell::new(ones())));
    // 1. compute gradient(s) of current operator wrt its input(s)
    // TODO from trait node.name->operator
    let op = ReLU {};
    // TODO avoid computing grad altogheter if var does not require it
    let op_inputs = node.borrow().variables.to_vec(); // TODO this does a copy!
    // let grads = op.backward(op_inputs, prev_grad.borrow_mut());
    let grads = foo(op_inputs, prev_grad);
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
    for (i, parent) in node.borrow().parents.as_ref().unwrap_or(&vec![]).iter().enumerate() {
        let g = Rc::clone(&grads[i]);
        backward_algo(Rc::clone(&parent), Some(g) );
    }
}

pub trait Operator {
    fn forward(&self, xs: Vec<Rc<RefCell<Tensor>>>) -> Tensor;
    fn backward(&self, xs: Vec<Rc<RefCell<Tensor>>>, grad: Rc<RefCell<Tensor>>)->Vec<Rc<RefCell<Tensor>>>;
    // fn accumulate_grad(&self, grad: Tensor) -> Tensor;
    // fn get_grad(&self) -> Tensor;
    fn attach_to_eager_graph(&self, xs: Vec<Rc<RefCell<Tensor>>>, op_output: &mut Tensor) {
        // keep references to operator inputs
        let mut vars: Vec<Rc<RefCell<Tensor>>> = Vec::new();
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
        let mut op: Rc<RefCell<Node>> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            Rc::new(RefCell::new(Node::new(
                OperatorNodes::ReLU,
                vars,
                Some(op_parents),
            )))
        } else {
            // first node in the graph
            Rc::new(RefCell::new(Node::new(OperatorNodes::ReLU, vars, None)))
        };
        
        // "attach" output var to graph 
        op_output.graph = Some(op);
    }
}

// TODO solve this variable ordering/naming problem
impl<'a> Operator for ReLU {
    fn backward(&self, xs: Vec<Rc<RefCell<Tensor>>>, grad: Rc<RefCell<Tensor>>)->Vec<Rc<RefCell<Tensor>>> {
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
    fn forward(&self, xs: Vec<Rc<RefCell<Tensor>>>) -> Tensor {
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

enum Operators {
    ReLU(ReLU),
}
enum OperatorNodes {
    ReLU,
}
