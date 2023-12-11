use num_traits::{Float, FromPrimitive};

use crate::operators;
use crate::tensor::init::calculate_fan_in_and_fan_out;
use crate::{
    operators::operators::{Operator, Operators},
    tensor::tensor::Tensor,
};

trait Layer<T: Float + FromPrimitive + 'static> {
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
            }];
            res = xs[0].clone();
        }
        res
    }
}

pub struct Identity;
pub struct Linear {
    w: Tensor<f32>, // quantized linear will be a different layer
    bias: Tensor<f32>,
}
impl Linear {
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // TODO optional bias
        // init from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        let w_shape = &[out_features, in_features];
        let (fan_in, _) = calculate_fan_in_and_fan_out(w_shape);
        let bound = 1. / f32::sqrt(fan_in as f32);

        Linear {
            w: Tensor::<f32>::kaiming_uniform(w_shape),
            bias: Tensor::<f32>::uniform(w_shape, -bound, bound),
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

    fn forward(&self, xs: Vec<Tensor<f32>>) -> Tensor<f32> {
        assert!(xs.len() == 1);
        let mut xwb = self.parameters();
        xwb.extend(xs);
        self.forward_default(xwb)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_linear(){
        todo!()
    }
}