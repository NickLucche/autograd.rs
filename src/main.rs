mod autograd;
use autograd::autograd::{Operator, Tensor, ReLU};
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let mut x = Tensor::new();
    let mut xs = vec![Rc::new(RefCell::new(x))];
    ReLU{}.forward(xs);
    println!("Hello, world!");
}
