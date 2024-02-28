use num_traits::{Float, FromPrimitive};

use crate::operators;
use crate::tensor::init::calculate_fan_in_and_fan_out;
use crate::{
    operators::operators::{Operator, Operators},
    tensor::tensor::Tensor,
};

pub trait Layer<T: Float + FromPrimitive + 'static> {
    // ops can either be lazily or eagerly initialized
    fn get_op_subgraph(&self) -> Vec<Operators>;
    fn parameters(&self) -> Vec<Tensor<T>>;
    fn forward(&self, xs: Vec<Tensor<T>>) -> Tensor<T> {
        self.forward_default(xs)
    }
    // layers only help in creating the graph, backward is called on tensors directly
    fn forward_default(&self, mut xs: Vec<Tensor<T>>) -> Tensor<T> {
        let operators = self.get_op_subgraph();
        // sequentially feed operations output into next operator
        // TODO forward with ref might be cleaner here
        let mut res: Tensor<T> = xs[0].clone();
        for op in operators {
            xs = vec![match op {
                Operators::ReLU(op) => op.forward(xs),
                Operators::Sigmoid(op) => op.forward(xs),
                Operators::MatMul(op) => op.forward(xs),
                Operators::Linear(op) => op.forward(xs),
                Operators::MeanSquaredError(op) => op.forward(xs),
                Operators::Mean(op) => op.forward(xs),
                Operators::Identity(op) => op.forward(xs),
            }];
            res = xs[0].clone();
        }
        res
    }
}

#[derive(Copy, Clone)]
pub struct Identity;
#[derive(Copy, Clone)]
pub struct ReLU;
#[derive(Copy, Clone)]
pub struct Sigmoid;
#[derive(Copy, Clone)]
pub struct MeanSquaredError;
#[derive(Copy, Clone)]
pub struct Conv2D;

impl<T: 'static> Layer<T> for Identity where T: Float+FromPrimitive{
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::Identity(operators::operators::Identity)]
    }
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}
#[derive(Clone)]
pub struct Linear {
    w: Tensor<f32>, // quantized linear will be a different layer
    bias: Tensor<f32>,
}
impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // TODO optional bias
        // init from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        let w_shape = &[in_features, out_features];
        let (fan_in, _) = calculate_fan_in_and_fan_out(w_shape);
        let bound = 1. / f32::sqrt(fan_in as f32);

        Linear {
            w: Tensor::<f32>::kaiming_uniform(w_shape),
            bias: Tensor::<f32>::uniform(&[1, out_features], -bound, bound),
        }
    }
}
impl Layer<f32> for Linear {
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::Linear(operators::operators::Linear)]
    }
    fn parameters(&self) -> Vec<Tensor<f32>> {
        vec![self.w.clone(), self.bias.clone()]
    }

    fn forward(&self, mut xs: Vec<Tensor<f32>>) -> Tensor<f32> {
        assert!(xs.len() == 1);
        xs.extend(self.parameters());
        self.forward_default(xs)
    }
}

impl<T: 'static> Layer<T> for ReLU where T: Float+FromPrimitive{
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::ReLU(operators::operators::ReLU)]
    }
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}
impl<T: 'static> Layer<T> for Sigmoid where T: Float+FromPrimitive{
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::Sigmoid(operators::operators::Sigmoid)]
    }
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}
impl<T: 'static> Layer<T> for MeanSquaredError where T: Float+FromPrimitive{
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::MeanSquaredError(operators::operators::MeanSquaredError)]
    }
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}
impl<T: 'static> Layer<T> for Conv2D where T: Float+FromPrimitive{
    fn get_op_subgraph(&self) -> Vec<Operators> {
        vec![Operators::MeanSquaredError(operators::operators::MeanSquaredError)]
    }
    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
    fn forward(&self, xs: Vec<Tensor<T>>) -> Tensor<T> {
        // check input is BCHW
        todo!()
    }
}





#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use crate::nn::layers::*;

    #[test]
    fn test_mlp(){
        let layer1 = Linear::new(20, 10, true);
        let relu = ReLU{};
        let layer2 = Linear::new(10, 1, true);
        let x = Tensor::<f32>::uniform(&[1, 20], -1.0, 1.0);

        let mut res = layer1.forward(vec![x.clone()]);
        res = relu.forward(vec![res]);
        res = layer2.forward(vec![res]);
        res.backward();
        assert!(*x.grad() != None);
    }
}