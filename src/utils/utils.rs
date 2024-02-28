pub mod export {
    use num_traits::{Float, FromPrimitive};
    use crate::autograd::autograd::Node;
    use crate::nn::model::NN;
    use crate::operators::operators::Operators;
    use crate::tensor::tensor::Tensor;

    const TORCH_IMPORT_TEMPLATE: &str = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n";

    fn autograd_tensor_to_torch_str<T: Float + FromPrimitive>(t: &Tensor<T>) -> String {
        // NOTE data is *not* copied, returns a placeholder tensor with same shape and dytpe
        // TODO
        let dtype = "torch.float32";
        format!("torch.zeros({:?}, dtype={})", t.shape(), dtype)
    }

    fn graph_to_pytorch_code<T: Float + FromPrimitive>(node: &Node<T>) -> String {
        // visit node, we embed op-specific logic here (e.g. number of expected input vars)
        let handle_activation = |activation_name: &str| {
            if !node.is_root_node() {
                // recursively call into inner op (child)
                let p = &node.parents[0].as_ref().unwrap().borrow();
                let inner_ops = graph_to_pytorch_code(&p);
                format!("F.{activation_name}({inner_ops})")
            } else {
                // there's no parents, must operate on single input var
                let t = autograd_tensor_to_torch_str(&node.variables[0]);
                format!("F.{activation_name}({t})")
            }
        };
        let torch_ops = match &node.operator {
            Operators::ReLU(op) => handle_activation("relu"),
            Operators::Sigmoid(op) => handle_activation("sigmoid"),
            Operators::MatMul(op) => {
                // any of the two inputs could come from another op or be a variable
                let mut ab = [String::from(""), String::from("")];
                for i in 0..2 {
                    if let Some(p) = &node.parents[i] {
                        // intermediate node, get subgraph definition for it
                        ab[i] = graph_to_pytorch_code(&p.borrow());
                    } else {
                        ab[i] = autograd_tensor_to_torch_str(&node.variables[i]);
                    }
                }
                let [a, b] = ab;
                return format!("torch.matmul({a}, {b})");
            },
            Operators::Linear(op) => {
                // NOTE here we assume W and b are parameters, hence they're leaf nodes
                // different from a standard MatMul + Add op
                let op_input = if node.is_root_node() {
                    autograd_tensor_to_torch_str(&node.variables[0])
                } else {
                    let p = &node.parents[0].as_ref().unwrap();
                    graph_to_pytorch_code(&p.borrow())
                };
                let x = &node.variables[0];
                let w = &node.variables[1];
                // let w = autograd_tensor_to_torch_str(&node.variables[1]);
                // let b = autograd_tensor_to_torch_str(&node.variables[2]);
                return format!("nn.Linear({}, {})({})", x.shape()[1], w.shape()[1], op_input);
            },
            Operators::MeanSquaredError(op) =>todo!(),
            Operators::Mean(op) =>todo!(),
            Operators::Identity(op) => {
                let inner = if !node.is_root_node() {
                    graph_to_pytorch_code(&node.parents[0].as_ref().unwrap().borrow())
                }else {
                    autograd_tensor_to_torch_str(&node.variables[0])
                };
                format!("nn.Identity()({inner})")
            },
        };
        torch_ops
    }

    pub fn to_torch_no_weights<T: Float + FromPrimitive + 'static>(model: &impl NN<T>, sample_input: Vec<Tensor<T>>)->String{
        // TODO tracing the model in this 'inlined' form will also trace weight initialization..
        // run through model to create graph
        let output = model.forward(sample_input);
        // walk graph to list ops in the model, mapping to Pytorch ops
        let graph = output.graph.unwrap();
        let torch_code = graph_to_pytorch_code(&graph.borrow());

        format!("{}{}", TORCH_IMPORT_TEMPLATE, torch_code)
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::layers::{Layer, Linear, ReLU, Identity};
    use crate::nn::model::NN;
    use crate::tensor::tensor::Tensor;
    use crate::utils::utils::export::to_torch_no_weights;
    use super::*;

    #[test]
    fn test_simple_mlp() {
        struct MyModel {layer1: Linear, relu: ReLU, layer2: Linear }
        impl MyModel {
            pub fn new()->Self {
                let layer1 = Linear::new(20, 10, true);
                let relu = crate::nn::layers::ReLU {};
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
                x = Identity{}.forward(vec![x]);
                self.layer2.forward(vec![x])
            }
        }
        let model = MyModel::new();
        let x = Tensor::<f32>::uniform(&[1, 20], -1.0, 1.0);

        let torch_code = to_torch_no_weights(&model, vec![x]);
        println!("{torch_code}");
    }
}