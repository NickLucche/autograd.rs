mod autograd;
mod tensor;
mod operators;

use operators::operators::{Operator, ReLU};
use tensor::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::prelude::*;

fn main() {
    let a = array![[1.,2.], [3., 4.]];
    let mut x = Tensor::from(a);
    x.data.view_mut().into_shape((4)).unwrap()[0] = 1.0;
    let mut xs = vec![Rc::new(RefCell::new(x))];
    let res = ReLU{}.forward(xs);
    for x in &res.data {
        print!("{}\t", x);
    }
}
