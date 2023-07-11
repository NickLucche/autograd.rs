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
fn backward_algo(node: Node, prev_grad: Option<Tensor>) {
    let prev_grad = prev_grad.unwrap_or(ones());
    // 1. compute gradient(s) of current operator wrt its input(s)
    // TODO from trait node.name->operator
    let op = ReLU {};
    // TODO avoid computing grad altogheter if var does not require it
    let grads = op.backward(node.variables, prev_grad);
    // 2. accumulate gradient on input vars
    for (i, var) in node.variables.iter().enumerate() {
        // var.borrow_mut().grad += grads[i];
        if var.borrow().requires_grad {
            let mut acc = var.borrow_mut().grad;
            for j in 0..acc.len() {
                acc[j] += grads[i].data[j];
            }
        }
    }
    // 3. recurse on parent nodes
    for parent in node.parents.unwrap_or(vec![]) {
        backward_algo(*parent.borrow(), Some(grads[0]));
    }
}
pub trait Operator {
    fn forward(&self, xs: Vec<Rc<RefCell<Tensor>>>) -> Tensor;
    fn backward(&self, xs: Vec<Rc<RefCell<Tensor>>>, grad: Tensor) -> Vec<Tensor>;
    // fn accumulate_grad(&self, grad: Tensor) -> Tensor;
    // fn get_grad(&self) -> Tensor;

    // fn get_parents(&self)->Vec<Tensor>;
}
// TODO solve this variable ordering/naming problem
impl<'a> Operator for ReLU {
    fn backward(&self, xs: Vec<Rc<RefCell<Tensor>>>, mut grad: Tensor) -> Vec<Tensor> {
        // having grad as input allows to avoid unnecessary allocations
        let x = xs[0].borrow();
        for i in 0..x.data.len() {
            if x.data[i] <= 0 {
                grad.data[i] = 0;
            }
        }
        vec![grad]
    }
    fn forward(&self, xs: Vec<Rc<RefCell<Tensor>>>) -> Tensor {
        // ****
        // TODO this code must be shared or called explicitely

        // keep references to operator inputs
        let mut vars: Vec<Rc<RefCell<Tensor>>> = Vec::new();
        let mut op_parents = Vec::new();
        for x in xs.iter() {
            vars.push(Rc::clone(&x));
            if !x.borrow().graph.is_none() {
                // input is the result of another op, attach current op to it in the graph
                op_parents.push(Rc::clone(x.borrow().graph.as_ref().unwrap()));
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

        // TODO we could attach it to out tensor if op not in_place, do if on that
        for x in &xs {
            if x.borrow().graph.is_none() {
                // init new graph!
                (*x).borrow_mut().graph = Some(Rc::clone(&op));
            }
        }
        // NOTE even when all input have graph, `op` can be reached with the backward chain as long as we have the latest node (the one we call backward from)!
        // If multiple final nodes/heads are present, multiple backward must be called (or just sum em like when you have multiple losses)

        // ****

        let mut t = Tensor::new();
        for (i, val) in xs[0].borrow_mut().data.iter().enumerate() {
            if *val > 0 {
                t.data[i] = *val;
            }
        }
        t
    }
}

enum Operators {
    ReLU(ReLU),
}
enum OperatorNodes {
    ReLU,
}
