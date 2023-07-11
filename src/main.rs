mod autograd;
mod tensor;
mod operators;

use operators::operators::{Operator, ReLU};
use tensor::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let mut x: Tensor = Tensor::new();
    x.data[0] = 69;
    let mut xs = vec![Rc::new(RefCell::new(x))];
    let res = ReLU{}.forward(xs);
    for x in &res.data {
        print!("{}\t", x);
    }
}
