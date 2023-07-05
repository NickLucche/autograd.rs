use std::ops;
use std::rc::Rc;
use std::cell::RefCell;

// TODO actual tensor lib for data
// TODO this thing might as well go on heap, since data itself does and use Box<Tensor> around
pub struct Tensor {
    requires_grad: bool,
    // TODO rename to graph_node
    // graph: Option<&'graph dyn Operator>
    // I want interior mutability to set only the graph ref, not the rest
    graph: Option< Rc<RefCell<Node>> >,
    // graph: Option< Rc<Node> >,
    data: Vec<i32>,
    grad: Vec<i32>,
}
impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            requires_grad: false,
            graph: None,
            data: vec![0; 10],
            grad: vec![0; 10]
        }
    }
    fn set_graph(&self, graph: Rc<RefCell<Node>>) {
        self.graph = Some(graph);
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
    pub variables: Vec< Rc<Tensor> >,
    pub parents: Option< Vec<Rc<Node>> >,
}

impl Node {
    fn new(
        name: OperatorNodes,
        variables: Vec< Rc<Tensor>>,
        parents: Option<Vec<Rc<Node>>>,
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
fn ones()->Tensor {
    Tensor::new()
}
// fn backward_algo(node: Node, grad: Option<Tensor>) {
//     let grad = grad.unwrap_or(ones());
//     // TODO from trait node.name->operator
//     let op = ReLU{};
//     op.backward(x)
//     for parent in node.parents {
        
//     }
// }
pub trait Operator {
    fn forward(&self, xs: Vec< Rc<Tensor> >) -> Tensor;
    fn backward(&self, x: &Tensor, grad: Tensor) -> Tensor;
    // fn accumulate_grad(&self, grad: Tensor) -> Tensor;
    // fn get_grad(&self) -> Tensor;

    // fn get_parents(&self)->Vec<Tensor>;
}

impl<'a> Operator for ReLU {
    fn backward(&self, x: &Tensor, mut grad: Tensor) -> Tensor {
        // having grad as input allows to avoid unnecessary allocations
        for i in 0..x.data.len() {
            if x.data[i] <= 0 {
                grad.data[i] = 0;
            }
        }
        grad
    }
    fn forward(&self, mut xs: Vec< Rc<Tensor> >) -> Tensor {
        // **** 
        // TODO this code must be shared or called explicitely

        // keep references to operator inputs
        let mut vars: Vec<Rc<Tensor>> = Vec::new();
        let mut op_parents = Vec::new();
        for x in xs.iter() {
            vars.push(Rc::clone(&x));
            if !x.graph.is_none() {
                // input is the result of another op, attach current op to it in the graph
                op_parents.push(Rc::clone(x.graph.as_ref().unwrap()));
            }
        }
        // instatiate new operator node on heap
        let mut op: Rc<Node> = if op_parents.len() > 0 {
            // attach to graph by linking its parents!
            Rc::new(Node::new(OperatorNodes::ReLU, vars, Some(op_parents)))
        } else {
            // first node in the graph
            Rc::new(Node::new(OperatorNodes::ReLU, vars, None))
        };

        // TODO we could attach it to out tensor if op not in_place, do if on that
        for x in xs.iter_mut() {
            if x.graph.is_none() {
                // init new graph!
                x.set_graph(RefCell::new(Rc::clone(op)));
                // x.graph = Some(Rc::clone(&op));
            }
        }
        // NOTE even when all input have graph, `op` can be reached with the backward chain as long as we have the latest node (the one we call backward from)!
        // If multiple final nodes/heads are present, multiple backward must be called (or just sum em like when you have multiple losses)

        // ****

        let mut t = Tensor::new();
        // for (i, val) in xs[0].data.iter().enumerate() {
        //     if *val > 0 {
        //         t.data[i] = *val;
        //     }
        // }
        t
    }
}

enum Operators {
    ReLU(ReLU),
}
enum OperatorNodes {
    ReLU,
}
