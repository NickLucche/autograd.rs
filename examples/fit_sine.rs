use std::f32::consts::PI;
use ndarray::stack;
use autograd_rs::autograd;
use autograd_rs::operators;
use autograd_rs::tensor;
use autograd_rs::nn;

use ndarray::prelude::*;
use num_traits::Pow;
use autograd_rs::nn::optim::Optimizer;
use tensor::tensor::Tensor;
use crate::nn::layers::{Layer, Linear, MeanSquaredError, ReLU, Sigmoid};
use crate::nn::optim::SGD;
use crate::nn::model::NN;

// https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#nn-module
fn main() {
    // fit y=sin(x) with a 3rd order polynomial with autograd.rs

    // generate random inputs and respective sin
    let x = Array::<f32, _>::linspace(-PI, PI, 2000);
    let y = x.mapv(|x| x.sin());
    // TODO handle shape (N,)->(N,1) or at least throw proper error
    let y = y.insert_axis(Axis(1));
    // we want to approximate our sin with the polynomial `y= a + b x + c x^2 + d x^3`,
    // hence we can learn the parameters a,b,c,d with a LinearLayer as:
    // [x, x^2, x^3]@[b,c,d] + a(bias) = y.
    // Let's create the input values x, x^2, x^3
    let mut xx = stack![Axis(1), x, x, x];
    for i in 0..xx.nrows() {
        xx[[i, 1]] = xx[[i, 1]].pow(2.0);
        xx[[i, 2]] = xx[[i, 2]].pow(3.0);
    }

    let x = Tensor::from(xx);
    let mut y = Tensor::from(y);
    y.requires_grad = false;

    struct MyModel {
        layer: Linear,
    }
    impl MyModel {
        pub fn new() -> Self {
            MyModel { layer: Linear::new(3, 1, true) }
        }
    }

    impl NN<f32> for MyModel {
        // cloning is "safe" because layers inner weights will still point to the same shared mem location
        fn layers(&self) -> Vec<Box<dyn Layer<f32>>> {
            vec![Box::new(self.layer.clone())]
        }
        fn forward(&self, xs: Vec<Tensor<f32>>) -> Tensor<f32> {
            self.layer.forward(xs)
        }
    }
    let model = MyModel::new();
    let mut params = model.parameters();

    // training params
    let lr = 1e-3;
    let n_epochs = 2000;
    let mut optim = SGD::new(&mut params, lr);
    for epoch in 0..n_epochs {
        let pred = model.forward(vec![x.clone()]);
        // NOTE Pytorch difference: 'sum' is used as reduction, which converges better
        let loss = MeanSquaredError {}.forward(vec![pred.clone(), y.clone()]);
        if epoch % 100 == 99 {
            println!("Epoch {epoch} || Pred {:?} || loss {:?}", &pred.shape(), loss.data());
        }
        // no need to zero_grad before backward(), grads are initialized
        loss.backward();

        // NOTE would need no_grad if tensor operations generated a graph here!
        // ** optimizer step **
        // {
        //     let bcd_params = &mut model.parameters()[0];
        //     let mut w = bcd_params.data_mut();
        //     let w_grad = bcd_params.grad();
        //     let w_grad_scaled = w_grad.as_ref().unwrap() * lr;
        //     *w -= &w_grad_scaled;
        //
        //     let a_bias = &mut model.parameters()[1];
        //     let mut b = a_bias.data_mut();
        //     let b_grad = a_bias.grad();
        //     let b_grad_scaled = b_grad.as_ref().unwrap() * lr;
        //     *b -= &b_grad_scaled;
        //  }
        // simplified to
        optim.step();

        model.zero_grad(); // or optim.zero_grad(&model);
    }
    let linear_layer = &model.parameters()[..];
    if let [w, bias] = linear_layer {
        let v = &w.data().into_raw_vec()[..];
        if let [b, c, d] = v {
            let a_bias = bias.data()[[0, 0]];
            println!("Result: y = {a_bias} + {b} x + {c} x^2 + {d} x^3");
        } else {}
    }
}
