use ndarray::linalg::Dot;
use ndarray::s;
use ndarray::{Array, ArrayD, Ix2, IxDyn};
use std::rc::Rc;

use crate::autograd::autograd::Node;
use crate::operators::functional_ndarray::{relu_backward_ndarray, relu_ndarray, im2col_ndarray};
use crate::storage_apply;
use crate::tensor::Primitive;
use crate::tensor::storage::{StorageType, CudaData};
use crate::tensor::tensor::{ones_like, ones_like_f32, zeros_like, Powi, Tensor};

use super::functional_ndarray::im2col_ndarray_backward;
use crate::utils::{SharedPtr, shared_ptr_new};



pub trait Operator {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
        Tensor<T>: Powi;

    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>>;

    // TODO should this function live in autograd? Self isn't even needed!
    fn attach_to_eager_graph<T: Primitive>(
        &self,
        mut inputs: Vec<Tensor<T>>,
        op_output: &mut Tensor<T>,
        operator: Operators,
    ) {
        // only fill in the parents of those input nodes generated from another Operator ("intermediate" ones)
        let mut op_parents = vec![None; inputs.len()];
        for (i, x) in inputs.iter_mut().enumerate() {
            if x.graph.is_some() {
                // input is the result of another op, attach current op to it in the graph (through "parents" nodes)
                op_parents[i] = Some(Rc::clone(x.graph.as_ref().unwrap()));
            }
            // drop previous references to the graph: `backwards` can only be called from latest output var (e.g. loss)!
            x.graph = None;
        }
        // instantiate new operator node on heap; first node in graph will have all entries in op_parents equal to None
        let op: SharedPtr<Node<T>> =
            shared_ptr_new(Node::new(operator, inputs.clone(), op_parents));

        // "attach" output var to graph
        op_output.graph = Some(op);
    }
}

pub struct Identity;

pub struct ReLU;

pub struct Sigmoid;

pub struct Softmax;

pub struct Mean;

pub struct MeanSquaredError;

pub struct Linear;

#[derive(Clone)]
pub struct Conv2D {
    in_channels: usize,
    out_channels: usize,
    k_size: usize,
    stride: usize,
    pad: usize,
}

pub struct MatMul;

pub struct Mul; // TODO elementwise

fn backward_dispatch(
    xs: Vec<Tensor<f32>>,
    grad: Tensor<f32>,
    back_f_cpu: impl Fn(Vec<&ArrayD<f32>>, &mut ArrayD<f32>) -> Vec<ArrayD<f32>>,
) -> Vec<Tensor<f32>> {
    let mut v_cpu: Vec<&ArrayD<f32>> = Vec::new();
    let mut v_cuda: Vec<&CudaData<f32>> = Vec::new();
    let xdata = xs[0].data();
    for x in &xs {
        if let StorageType::ArrayData(arr) = &*xdata {
            v_cpu.push(&arr);
        } else if let StorageType::CudaData(arr) = &*xdata {
            v_cuda.push(&arr);
        }
    }
    if v_cpu.len() < xs.len() && v_cuda.len() < xs.len() {
        panic!("Tensors must be on same device!");
    }
    let mut gstorage = grad.data_mut();
    match &mut *gstorage {
        StorageType::ArrayData(grad_arr) => {
            let grads_v = back_f_cpu(v_cpu, grad_arr);
            grads_v.into_iter().map(|g| Tensor::from(g)).collect()
        }
        StorageType::CudaData(grad_cuda) => todo!(),
        _ => panic!("Tensors must be on same device"), // TODO return proper result
    }
}

// TODO solve this variable ordering/naming problem
impl Operator for ReLU {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T> {
        // // TODO in_place: treat x as output and attach it to current op
        let mut t = Tensor::from(storage_apply!(
            &*xs[0].data(),
            relu_ndarray,
            |x: &CudaData<T>| todo!()
        ));
        self.attach_to_eager_graph(xs, &mut t, Operators::ReLU(ReLU));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        // let x = xs.get(0).unwrap();
        // // re-use grad storage
        // for (g_i, x_i) in grad.data_mut().iter_mut().zip(x.data().iter()) {
        //     if *x_i <= 0.0 {
        //         *g_i = 0.0;
        //     }
        // }
        backward_dispatch(xs, grad, relu_backward_ndarray)
    }
}

impl Operator for Linear {
    /**
     * x @ W + b
     */
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let x: &Tensor<T> = &xs[0];
        let w: &Tensor<T> = &xs[1];
        let b: &Tensor<T> = &xs[2];
        // x.dot creates new tensor, +b: &Tensor adds b to it in-place
        let mut t = x.dot(w) + b;
        self.attach_to_eager_graph(xs, &mut t, Operators::Linear(Linear));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        // if confused->https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/fc_layer

        // b and W will surely need grad (NOTE non learned bias not supported)
        let x: &Tensor<f32> = &xs[0];
        let w: &Tensor<f32> = &xs[1];
        // let b: &Tensor<f32> = &xs[2].borrow();
        let g: &Tensor<f32> = &grad;

        // NOTE in the backward pass, since we need to compute grads as f32 (dot runs with float only),
        // we also need the weights to be f32. In the forward pass (e.g. inference), we can experiment with int only ops
        let dx = g.dot(&w.t_clone()); // TODO handle transpose with tensorview
        let dw = x.t_clone().dot(g);
        let db = g.sum_axis(0);
        vec![dx, dw, db]
    }
}

impl Operator for MatMul {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        // TODO handle higher dims https://pytorch.org/docs/stable/generated/torch.bmm.html
        let x: &Tensor<T> = &xs[0];
        let w: &Tensor<T> = &xs[1];
        let mut t = x.dot(w);
        self.attach_to_eager_graph(xs, &mut t, Operators::MatMul(MatMul));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let x: &Tensor<f32> = &xs[0];
        let w: &Tensor<f32> = &xs[1];
        let g: &Tensor<f32> = &grad;

        let dx = g.dot(w);
        let dw = g.t_clone().dot(x);
        vec![dx, dw]
    }
}

impl Sigmoid {
    pub fn sigmoid_inplace<T: Primitive>(x: &Tensor<T>) -> &Tensor<T> {
        x.data_mut()
            .mapv_inplace(|x| T::from(1.0 / (1.0 + f32::exp(-x.to_f32().unwrap()))).unwrap());
        x
    }
    pub fn sigmoid<T: Primitive>(x: &Tensor<T>) -> Tensor<T> {
        let t = x
            .data_mut()
            .mapv(|x| T::from(1.0 / (1.0 + f32::exp(-x.to_f32().unwrap()))).unwrap());
        Tensor::from(t)
    }
}

impl Operator for Sigmoid {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T> {
        let x: &Tensor<T> = &xs[0];
        // compute sigmoid in f32, then convert value back to T (avoiding `as_type` copy)
        // |x| 1.0/(1.0 + f64::exp(-x))
        // TODO inplace version, use x here
        let mut t = Sigmoid::sigmoid(&x);

        self.attach_to_eager_graph(xs, &mut t, Operators::Sigmoid(Sigmoid));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let g: &Tensor<f32> = &grad;
        // sig'(x) = sig(x) * (1 - sig(x)), \frac{d}{dx} \sigma(x) = \sigma(x) \cdot (1 - \sigma(x))
        // TODO re-use previous result, requires storing ref to "t" above and some api changes
        // do the following only when inplace is set
        let x: &Tensor<f32> = &xs[0];
        let sig = Sigmoid::sigmoid(x);
        let dx = (1.0 - &sig) * sig;
        // Tensor * &Tensor, in-place
        vec![dx * g]
    }
}

/// Computes np.mean((prediction - target) ** 2), which is Pytorch default (reduction='mean'),
/// with batch dim (0) assumed to be present.
impl Operator for MeanSquaredError {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
        Tensor<T>: Powi,
    {
        let pred: &Tensor<T> = &xs[0];
        let target: &Tensor<T> = &xs[1];
        let mut t = (pred - target).powi_inplace(2).mean(None);
        t.reshape(&[1, 1]); // mean reduces (rightly) to a scalar, we need 2d
                            // TODO macro for this?
        self.attach_to_eager_graph(xs, &mut t, Operators::MeanSquaredError(MeanSquaredError));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let pred: &Tensor<f32> = &xs[0];
        let target: &Tensor<f32> = &xs[1];
        // batch dim
        let B = pred.shape()[0] as f32;
        // g *= 2 / (shape[0])
        let dpred = (pred - target) * 2.0 / B;
        // self.parents[0].backward(g * (self.parents[0].value - self.parents[1].value)
        // TODO target needs no grad, we can just return one value and let backward_algo zip
        // loop over the shortest array (only works if not required var is last,e.g. zip([x1, x2], [g1]))
        vec![dpred * &grad]
    }
}

impl Operator for Mean {
    // TODO axis as Mean attribute
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let mut t = xs[0].mean(None);
        self.attach_to_eager_graph(xs, &mut t, Operators::Mean(Mean));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let x = &xs[0];
        let t = ones_like(x) * 1.0 / x.size() as f32;
        vec![t * &grad]
    }
}

impl Operator for Identity {
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        let mut t = xs[0].clone();
        self.attach_to_eager_graph(xs, &mut t, Operators::Identity(Identity));
        t
    }
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        vec![grad]
    }
}

impl From<&Conv2D> for Conv2D {
    fn from(other: &Conv2D) -> Self {
        Conv2D {
            in_channels: other.in_channels,
            out_channels: other.out_channels,
            k_size: other.k_size,
            stride: other.stride,
            pad: other.pad,
        }
    }
}
impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        k_size: usize,
        stride: usize,
        pad: usize,
    ) -> Self {
        Conv2D {
            in_channels: in_channels,
            out_channels: out_channels,
            k_size: k_size,
            stride: stride,
            pad: pad,
        }
    }

    fn get_output_shape(&self, h: usize, w: usize) -> [usize; 2] {
        let new_height = (h - self.k_size + 2 * self.pad) / self.stride + 1;
        let new_width = (w - self.k_size + 2 * self.pad) / self.stride + 1;
        [new_height, new_width]
    }
    /// Lay out data in an accelerator-friendly way (cpus should also be happy with AVX and similar)
    /// by "unfolding" the different patches of the input image that the kernel slides over, so that
    /// we can resort to a (batched) matmul.
    pub fn im2col<T: Primitive>(
        im: &Tensor<T>,
        k_size: usize,
        stride: usize,
        pad: usize,
    ) -> Result<ArrayD<T>, String> {
        storage_apply!(&*im.data(), |x: &ArrayD<T>| im2col_ndarray(x, k_size, stride, pad), |x: &CudaData<T>| todo!())
    }

    // Computes backward grads for the im2col op. It accumulates gradients on overlapping strides.
    pub fn im2col_backward<T: Primitive>(
        col: &Tensor<T>,
        h_out: usize,
        w_out: usize,
        stride: usize,
        k_size: usize,
    ) -> Result<ArrayD<T>, String> {
        storage_apply!(&*col.data(), |x: &ArrayD<T>| im2col_ndarray_backward(x, h_out, w_out, stride, k_size), |x: &CudaData<T>| todo!())
    }
}

impl Operator for Conv2D {
    ///
    /// # Arguments
    ///
    /// * `xs`: [0] input activation map BxCxHxW, [1] kernels ("compressed") 1xKxKxC_out [2] bias 1xC_out.
    fn forward<T: Primitive + 'static>(&self, xs: Vec<Tensor<T>>) -> Tensor<T>
    where
        Array<T, Ix2>: Dot<Array<T, Ix2>, Output = Array<T, Ix2>>,
    {
        // great ref https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster#forward-graph
        let im = &xs[0];
        // 1xKxKxC_out
        let filters = &xs[1];
        let bias = &xs[2]; // 1xC_out, one value per filter
                           // BxPxKKC
        let mut col = Tensor::from(Conv2D::im2col(im, self.k_size, self.stride, self.pad).unwrap());
        // reshape filters to prepare for matmul
        let kernel = filters.data();
        // do the actual filter broadcast here, lazily -> C_inxKxKxC_out, owned due to into_shape cont memory req >.<
        let kernel = kernel.broadcast(IxDyn(&[
            self.in_channels,
            self.k_size,
            self.k_size,
            self.out_channels,
        ]));
        let kkc = self.k_size * self.k_size * self.in_channels;
        // TODO would like to avoid this into_owned -> totensor thing, add broadcast op like reshape
        let kernel = kernel
            .into_shape((kkc, self.out_channels))
            .unwrap()
            .into_owned();
        let kernel = Tensor::from(kernel);
        // tmp trick for bmm to get 2d matrix | B*PxKKC - KKCxC_out -> B*PxC_out
        col.reshape(&[col.shapei(0) * col.shapei(1), kkc]);
        let mut t = col.dot(&kernel) + bias;
        // reshape conv output to match expected shape
        let [h_out, w_out] = self.get_output_shape(im.shapei(2), im.shapei(3));
        // TODO swap axis?
        t.reshape(&[im.shapei(0), h_out, w_out, self.out_channels]);

        // cached for backward pass (computed at f32)
        let mut xs = xs.clone();
        xs.push(col);
        self.attach_to_eager_graph(xs, &mut t, Operators::Conv2D(Conv2D::from(self)));
        t
    }
    ///
    /// # Arguments
    ///
    /// * `xs`: [0] im BxCxHxW, [1] kernels ("compressed") 1xKxKxC_out [2] bias 1xC_out [3] im2col result B*PxKKC.
    /// * `grad`: same shape as conv out, BxH_outxW_outxC_out
    fn backward(&self, xs: Vec<Tensor<f32>>, grad: Tensor<f32>) -> Vec<Tensor<f32>> {
        let im: &Tensor<f32> = &xs[0];
        let filters: &Tensor<f32> = &xs[1];
        let g: &Tensor<f32> = &grad;

        // B*PxC_out <- BxH_outxW_outxC_out
        let mut g = g.clone();
        g.reshape(&[g.shapei(0) * g.shapei(1) * g.shapei(2), g.shapei(3)]);
        // sum contributions over PB dim (broadcasted at runtime), since you have one value per kernel (Cout)
        let db = g.sum_axis(0);

        // B*PxC_out @ 1xKxKxC_out
        let kkc = self.k_size * self.k_size * self.in_channels;
        // 1xKxKxC_out->CxKxKxC_out
        let kernels = filters.clone();
        let karr = kernels.data_mut();
        let karr = karr.broadcast(IxDyn(&[
            self.in_channels,
            self.k_size,
            self.k_size,
            self.out_channels,
        ]));
        let mut kernels = Tensor::from(karr);
        kernels.reshape(&[kkc, self.out_channels]);

        // B*PxC_out @ C_outxKKC -> B*PxKKC
        let dx = g.dot(kernels.t()).unsqueeze(0);
        let [h_out, w_out] = self.get_output_shape(im.shapei(2), im.shapei(3));
        let dx = Tensor::from(
            Conv2D::im2col_backward(&dx, h_out, w_out, self.stride, self.k_size).unwrap(),
        );
        // TODO remove padding around edges on dx
        // bring it back to original shape
        // TODO needed if we remove into_owned in broadcast op
        // kernels.swap_axes(0, -1);
        // kernels.reshape(&[1, self.k_size, self.k_size, self.in_channels, self.out_channels]);

        // just like Linear, but matrix 'col' is derived from inputs->we cache it during forward and re-use it here
        // torch employs a "ctx.save_for_backward" general api, we're not that fancy here
        let mut col = xs[3].clone();
        // must sum contributions on 'C' dimension due to broadcasting to update kernels!
        // BPxKKC->BPxKK
        col.reshape(&[
            col.shapei(0),
            col.shapei(1) / self.in_channels,
            self.in_channels,
        ]);
        col = col.sum_axis(-1);
        // KKCxB*P @ B*PxC_out  -> KKCxC_out (filters)
        let mut dw = col.t().dot(&g);
        dw.reshape(&[self.k_size, self.k_size, self.out_channels]);

        vec![dx, dw, db]
    }
}

// took a while to figure out a way to deal with dispatching and object safety violated due to generics in traits
// see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=504f29b8003d70964caee607b5655a88
// thanks to https://www.possiblerust.com/pattern/3-things-to-try-when-you-can-t-make-a-trait-object#code-1
pub enum Operators {
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Linear(Linear),
    MatMul(MatMul),
    MeanSquaredError(MeanSquaredError),
    Mean(Mean),
    Identity(Identity),
    Conv2D(Conv2D),
}

impl Into<String> for Operators {
    fn into(self) -> String {
        match self {
            Operators::ReLU(_) => String::from("ReLU"),
            Operators::Sigmoid(_) => String::from("Sigmoid"),
            Operators::Linear(_) => String::from("Linear"),
            Operators::MatMul(_) => String::from("MatMul"),
            Operators::MeanSquaredError(_) => String::from("MeanSquaredError"),
            Operators::Mean(_) => String::from("Mean"),
            Operators::Identity(_) => String::from("Identity"),
            Operators::Conv2D(_) => String::from("Conv2D"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_relu_on_mat() {
        let mut a = array![[-1., -2.], [3., 4.]];
        let b = array![[-1., -2.], [3., 4.]];
        a += &b;
        let x = Tensor::from(a);
        x.data_mut().view_mut().into_shape(4).unwrap()[0] = 1.0;
        assert!(x.data()[[0, 0]] == 1.0);

        let xs = vec![x];
        let res = ReLU {}.forward(xs);
        for x in res.data().iter() {
            print!("{}\t", x);
        }
        assert!(res.requires_grad);

        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[1., 0.,], [6., 8.]]
        );
    }

    #[test]
    fn test_linear() {
        let w = Tensor::from(array![[1., 1.], [1., 1.]]);
        let b = Tensor::from(array![[1., 1.]]);
        let x = Tensor::from(array![[1., 1.]]);
        let linear = Linear {};
        let xs = vec![x, w, b];
        let res = linear.forward(xs);
        println!("{:?}", res.data);
        assert_eq!(
            res.data().view().into_dimensionality::<Ix2>().unwrap(),
            array![[3., 3.,]]
        );
    }

    #[test]
    fn test_im2col() {
        let nums = Array::range(1.0, 49.0, 1.0);
        let a = nums.into_shape((1, 3, 4, 4)).unwrap();
        let x = Tensor::from(a);
        let res = Conv2D::im2col(&x, 2, 1, 0).unwrap();

        let expected = array![
            [1., 2., 3., 5., 6., 7., 9., 10., 11.],
            [2., 3., 4., 6., 7., 8., 10., 11., 12.],
            [5., 6., 7., 9., 10., 11., 13., 14., 15.],
            [6., 7., 8., 10., 11., 12., 14., 15., 16.],
            [17., 18., 19., 21., 22., 23., 25., 26., 27.],
            [18., 19., 20., 22., 23., 24., 26., 27., 28.],
            [21., 22., 23., 25., 26., 27., 29., 30., 31.],
            [22., 23., 24., 26., 27., 28., 30., 31., 32.],
            [33., 34., 35., 37., 38., 39., 41., 42., 43.],
            [34., 35., 36., 38., 39., 40., 42., 43., 44.],
            [37., 38., 39., 41., 42., 43., 45., 46., 47.],
            [38., 39., 40., 42., 43., 44., 46., 47., 48.],
        ];
        let expected = expected.t().into_shape((1, 9, 12)).unwrap().into_dyn();
        assert_eq!(res, expected);
        // with padding
        let nums = Array::range(1.0, 9.0, 1.0);
        let a = nums.into_shape((1, 2, 2, 2)).unwrap();
        let x = Tensor::from(a);
        let res = Conv2D::im2col(&x, 2, 1, 1).unwrap();

        // 9x4*2 (#slides X k*k*C)
        let expected = array![
            [0., 0., 0., 1., 0., 0., 0., 5.],
            [0., 0., 1., 2., 0., 0., 5., 6.],
            [0., 0., 2., 0., 0., 0., 6., 0.],
            [0., 1., 0., 3., 0., 5., 0., 7.],
            [1., 2., 3., 4., 5., 6., 7., 8.],
            [2., 0., 4., 0., 6., 0., 8., 0.],
            [0., 3., 0., 0., 0., 7., 0., 0.],
            [3., 4., 0., 0., 7., 8., 0., 0.],
            [4., 0., 0., 0., 8., 0., 0., 0.],
        ];
        let expected = expected.into_shape((1, 9, 8)).unwrap().into_dyn();
        assert_eq!(res, expected);
    }

    #[test]
    fn test_conv2d() {
        // with a Layer we get properly initialized kernels, here we just make up our own
        let num_kernels = 3;
        let kernel = ArrayD::<f32>::ones(IxDyn(&[2, 2, num_kernels]));
        let bias = ArrayD::<f32>::ones(IxDyn(&[num_kernels]));
        let im = ArrayD::<f32>::ones(IxDyn(&[1, 2, 4, 4]));
        let conv = Conv2D {
            in_channels: 2,
            out_channels: num_kernels,
            k_size: 2,
            stride: 1,
            pad: 0,
        };
        let mut xs: Vec<Tensor<f32>> = vec![im, kernel, bias]
            .into_iter()
            .map(|x| Tensor::from(x))
            .collect();
        let res = conv.forward(xs.clone());
        let expected = ArrayD::from_elem(IxDyn(&[1, 3, 3, num_kernels]), 9.0);
        // println!("res shape {:?}", res.shape());
        // println!("res {:?}", res.data());
        assert_eq!(&*res.data(), expected);

        let res = Tensor::from(Conv2D::im2col(&xs[0], 2, 1, 0).unwrap());
        // simulate B*P layout that we expect with a squeeze since batch is 1
        let res = res.squeeze();
        xs.push(res);
        let [ho, wo] = conv.get_output_shape(4, 4);
        let grad = Tensor::ones(&[1, ho, wo, num_kernels]);
        let grads = conv.backward(xs.clone(), grad);
        for i in 0..3 {
            assert_eq!(grads[i].shape(), xs[i].shape());
        }
    }
}
