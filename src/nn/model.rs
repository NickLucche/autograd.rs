use num_traits::{Float, FromPrimitive};
use crate::tensor::tensor::Tensor;
use super::layers::Layer;

// this could also be a trait
// you have to define the forward on your own
pub trait NN<T: Float+FromPrimitive+'static>{
    fn layers(&self)->Vec<Box<dyn Layer<T>>>; // TODO experimental use of Box, may have to change for API
    fn parameters(&self)->Vec<Tensor<T>> {
        // accumulate parameters and return them so that optimizer can use them
        self.layers().iter().flat_map(|l| l.parameters()).collect()
    }
    fn forward(&self, xs: Vec<Tensor<T>>)->Tensor<T>;

}


#[cfg(test)]
mod tests {
    use crate::nn::layers::{Linear, ReLU};
    use super::*;

    #[test]
    fn test_mlp(){
        struct MyModel {layer1: Linear, relu: ReLU, layer2: Linear };
        impl MyModel {
            pub fn new()->Self {
                let layer1 = Linear::new(20, 10, true);
                let relu = ReLU {};
                let layer2 = Linear::new(10, 1, true);
                MyModel {layer1, relu, layer2}
            }
        }
        impl NN<f32> for MyModel {

            fn layers(&self) -> Vec<Box<dyn Layer<f32>>> {
                vec![Box::new(self.layer1.clone()), Box::new(self.relu), Box::new(self.layer2.clone())]
            }
            fn forward(&self, xs: Vec<Tensor<f32>>) -> Tensor<f32> {
                let mut x = self.layer1.forward(xs);
                x = self.relu.forward(vec![x]);
                self.layer2.forward(vec![x])
            }
        }
        let model = MyModel::new();
        let x = Tensor::<f32>::uniform(&[1, 20], -1.0, 1.0);
        let res = model.forward(vec![x.clone()]);
        assert!(res.data().shape() == &[1,1]);
        assert!(*res.grad()==None);
    }
}
