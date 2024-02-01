use autograd_rs::autograd;
use autograd_rs::operators;
use autograd_rs::tensor;
use autograd_rs::nn;

use ndarray::prelude::*;
use tensor::tensor::Tensor;
use crate::nn::layers::{Layer, Linear, MeanSquaredError, ReLU, Sigmoid};
use crate::nn::model::NN;
use crate::operators::operators::shared_ptr_new;

fn main() {
    struct MyModel {layer1: Linear, sig: Sigmoid, layer2: Linear }
    impl MyModel {
        pub fn new()->Self {
            let layer1 = Linear::new(784, 128, true);
            let sig = Sigmoid {};
            let layer2 = Linear::new(128, 10, true);
            MyModel {layer1, sig, layer2}
        }
    }
    impl NN<f32> for MyModel {

        fn layers(&self) -> Vec<Box<dyn Layer<f32>>> {
            vec![Box::new(self.layer1.clone()), Box::new(self.sig), Box::new(self.layer2.clone())]
        }
        fn forward(&self, xs: Vec<Tensor<f32>>) -> Tensor<f32> {
            // TODO sequential would help here
            let mut x = self.layer1.forward(xs);
            x = self.sig.forward(vec![x]);
            self.layer2.forward(vec![x])
        }
    }
    let model = MyModel::new();

    // data (to be abstracted away)
    let x_train = Array::<f32, _>::zeros([16, 784]);
    let y_train = Array::<f32, _>::zeros([16, 10]);

    // training params
    let lr = 0.1f32;
    let n_epochs = 10;
    let batch_size = 8;
    for epoch in 0..n_epochs {
        // go through batches
        for i in (0..16).step_by(batch_size) {
            let x = x_train.slice(s![i..i+batch_size, ..]);
            let y = y_train.slice(s![i..i+batch_size, ..]);
            // TODO tensor from array view
            let x = Tensor::from(x.to_owned());
            let mut y = Tensor::from(y.to_owned());
            y.requires_grad = false;

            let pred = model.forward(vec![x]);
            let loss = MeanSquaredError{}.forward(vec![pred.clone(), y.clone()]);
            println!("Epoch {epoch} || Pred {:?} || loss {:?}", &pred.shape(), loss.data());
            assert_eq!(*y.grad(), None); // no grad is computed for y

            loss.backward();

            // NOTE would need no_grad if tensor operations generated a graph!
            // ** optimizer step **
            for p in model.parameters().iter_mut() {
                let new_p = &*p.data() - (p.grad().as_ref().unwrap() * lr);
                p.data = shared_ptr_new(new_p);
                // println!("Epoch {epoch} || Param {:?} || New value mean {:?}", p.shape(), p.mean(None).data()[0]);
                println!("Epoch {epoch} || Param {:?} ", p.shape());
            }
            model.zero_grad();
        }
    }

}
