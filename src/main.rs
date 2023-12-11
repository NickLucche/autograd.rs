mod autograd;
mod operators;
mod tensor;
mod nn;

use ndarray::prelude::*;
use operators::operators::{Operator, ReLU};
use std::cell::RefCell;
use std::rc::Rc;
use tensor::tensor::Tensor;

fn main() {
    let a = array![[1., 2.], [3., 4.]];
    let mut x = Tensor::from(a);
    x.data_mut().view_mut().into_shape(4).unwrap()[0] = 1.0;
    let xs = vec![x];
    let res = ReLU {}.forward(xs);
    for x in res.data().iter() {
        print!("{}\t", x);
    }
}
