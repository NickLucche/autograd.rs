use ndarray::ArrayD;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::tensor::Tensor;

/**
 * From Pytorch https://github.com/pytorch/pytorch/blob/8c1567d021e717a74d19e0b96efdb69df3265826/torch/nn/init.py#L407
 * Fill the input `Tensor` with values using a Kaiming uniform distribution
 *
 * \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
 *
 * Also known as He initialization
 *
 * Simplified torch.nn.init.calculate_gain by assuming LeakyReLU with
 * a=sqrt(5) (from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear).
 *
 */
pub fn kaiming_uniform(tensor_shape: &[usize]) -> Tensor<f32> {
    let negative_slope = f32::sqrt(5.0);
    let gain = f32::sqrt(2.0 / (1. + f32::powi(negative_slope, 2)));
    let (fan_in, _) = calculate_fan_in_and_fan_out(tensor_shape);
    let std = gain / f32::sqrt(fan_in as f32);

    let bound = f32::sqrt(3.0) * std;

    Tensor::from(ArrayD::random(tensor_shape, Uniform::new(-bound, bound)))
}

pub fn calculate_fan_in_and_fan_out(tensor_shape: &[usize]) -> (usize, usize) {
    assert!(tensor_shape.len() >= 2);

    let num_input_fmaps = tensor_shape[1];
    let num_output_fmaps = tensor_shape[0];
    let mut receptive_field_size = 1;
    if tensor_shape.len() > 2 {
        for s in &tensor_shape[2..] {
            receptive_field_size *= s
        }
    }
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;

    (fan_in, fan_out)
}

pub fn uniform(tensor_shape: &[usize], low: f32, high: f32) -> Tensor<f32> {
    Tensor::from(ArrayD::random(tensor_shape, Uniform::new(low, high)))
}
