use crate::nn::model::NN;
use crate::tensor::tensor::Tensor;

pub trait Optimizer {
    // for training, we assume f32 weights here
    fn step(&mut self);
    fn zero_grad(&self, model: &impl NN<f32>) {
        // here to mimic Pytorch API
        model.zero_grad()
    }
}

pub struct SGD<'a> {
    parameters: &'a mut Vec<Tensor<f32>>,
    lr: f32,
}

impl<'a> SGD<'a> {
    pub fn new(parameters: &'a mut Vec<Tensor<f32>>, lr: f32) -> Self {
        SGD { parameters, lr }
    }
}

impl<'a> Optimizer for SGD<'a> {
    fn step(&mut self) {
        for p in self.parameters.iter_mut() {
            // update weight "memory location" so that all clones will have updated weights
            let mut w = p.data_mut();
            let w_grad = p.grad();
            let w_grad_scaled = w_grad.as_ref().unwrap() * self.lr;
            // Add/Sub-Assign only implemented for &Array, we need ref here
            *w -= &w_grad_scaled;
        }
    }
}

