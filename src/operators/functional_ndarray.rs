use std::vec;

// Implementation of operators (ReLU, Linear..) which make use of the ndarray interface
// directly; this is to avoid having a joint interface on StorageType for cuda
// and ndarray for operations that wouldn't make sense on cuda (ie slow iters).
use ndarray::linalg::Dot;
use ndarray::s;
use ndarray::{Array, ArrayD, Ix2, IxDyn};
use crate::tensor::Primitive;

// NOTE if there's some function you dont want to implement in cuda(ie broadcast)
// just move its implementation here and avoid common interface

pub fn relu_ndarray<T: Primitive>(x: &ArrayD<T>) -> ArrayD<T> {
    let mut t = ArrayD::<T>::zeros(x.raw_dim());
    for (tv, xv) in t.iter_mut().zip(x.iter()) {
        if *xv > T::from(0).unwrap() {
            *tv = *xv;
        }
    }
    t
}
pub fn relu_backward_ndarray(xs: Vec<&ArrayD<f32>>, grad: &mut ArrayD<f32>) -> Vec<ArrayD<f32>> {
    let x = xs[0];
    // re-use grad storage: avoid optimization for now
    // for (g_i, x_i) in grad.iter_mut().zip(x.iter()) {
    let mut g = grad.clone();
    for (g_i, x_i) in g.iter_mut().zip(x.iter()) {
        if *x_i <= 0.0 {
            *g_i = 0.0;
        }
    }
    vec![g]
}


/**
 * See operators.Conv2d.im2col.
 */
pub fn im2col_ndarray<T: Primitive>(
    im: &ArrayD<T>,
    k_size: usize,
    stride: usize,
    pad: usize,
) -> Result<ArrayD<T>, String> {
    // single ksize for both x/y direction for now
    if let [b, c, h, w] = im.shape()[..] {
        // number of conv operations/slides on both axis
        let new_height = (h - k_size + 2 * pad) / stride + 1;
        let new_width = (w - k_size + 2 * pad) / stride + 1;
        // #conv_ops X kernel_size*C, ie each conv is unfolded (channel-wise too)
        let mut col =
            ArrayD::<T>::zeros(IxDyn(&[b, new_height * new_width, c * k_size * k_size]));

        // since reshaping (below) requires a contiguous array, we re-use the same memory location so we save allocation
        // also, handling lateral im-filter overlaps when padding is tricky
        let mut patch_buffer = ArrayD::<T>::zeros(IxDyn(&[b, c * k_size * k_size]));
        let k_size = k_size as i32; // could overflow below!
        let pad = pad as i32;
        // go over each kernel slide (nh*nw="#convolutions") NOT over "col" pixels
        for y in 0..new_height {
            // NOTE we don't actually pad the input as it can be very expensive; we can recognize when patch
            // is out of bound (due to padding, wrt im origin) and only copy the sub-slice we need
            let patch_y = (y * stride) as i32 - pad;
            for x in 0..new_width {
                // lay out patch as a column vector
                let patch_x = (x * stride) as i32 - pad;
                // how you would do it if there was no padding, select patch and flatten
                // let patch_idxs = s![.., .., patch_y..patch_y+k_size, patch_x..patch_x+k_size];
                // let patch = im.slice(patch_idxs).into_owned();
                // let patch = patch.into_shape((b, patch.len() / b)).unwrap();
                // ..in particular, we handle the tricky case ('|' im boundaries, '/'kernel boundaries)
                // |X../X|0/
                // |X../X|0/   where X-0-X-0 slice needs to be flattened in this order

                let mut counter = 0;
                for ch in 0..c {
                    for i in patch_y..(patch_y + k_size) {
                        for j in patch_x..(patch_x + k_size) {
                            if i >= 0 && i < h as i32 && j >= 0 && j < w as i32 {
                                patch_buffer.slice_mut(s![.., counter]).assign(&im.slice(s![
                                    ..,
                                    ch,
                                    i,
                                    j
                                ]));
                            }
                            counter += 1;
                        }
                    }
                }
                // this assigns a whole row of "col" (data locality happy), indexing from patch y/x coordinates
                col.slice_mut(s![.., y * new_width + x, ..])
                    .assign(&patch_buffer);
                patch_buffer.fill(T::zero());
            }
        }
        return Ok(col);
    }
    Err(format!(
        "Expected image of shape BCHW, got shape {:?}",
        im.shape()
    ))
}


pub fn im2col_ndarray_backward<T: Primitive>(
    col: &ArrayD<T>,
    h_out: usize,
    w_out: usize,
    stride: usize,
    k_size: usize,
) -> Result<ArrayD<T>, String> {
    if let [b, P, K] = col.shape()[..] {
        // original image sizes
        let c = K / (k_size * k_size);
        let h = (h_out - 1) * stride + k_size;
        let w = (w_out - 1) * stride + k_size;
        let mut im = ArrayD::<T>::zeros(IxDyn(&[b, c, h, w]));
        // go through each flattened patch, reshape it and sum contributions on overlapping patches
        for i in 0..P {
            // BxK*K*C
            let row = col.slice(s![.., i, ..]);
            let y = (i / w_out) * stride;
            let x = (i % w_out) * stride;
            let mut patch = im.slice_mut(s![.., .., y..y + k_size, x..x + k_size]);
            // patch += &row; needs extra signature..
            // we can reshape without owning view as the slice is contiguous
            let p = &patch + &row.into_shape((b, c, k_size, k_size)).unwrap();
            patch.assign(&p);
        }
        return Ok(im);
    }
    Err(format!(
        "Expected tensor of shape BxH_out*W_outxK*K*C, got shape {:?}",
        col.shape()
    ))
}