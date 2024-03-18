use crate::tensor::tensor::{Primitive, Tensor};
use ndarray::{stack, ArrayD};
use ndarray_rand::rand::prelude::SliceRandom;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;

pub trait Dataset<T: Primitive, U: Primitive> {
    fn get_item(&self, idx: usize) -> (ArrayD<T>, ArrayD<U>);
    fn len(&self) -> usize;
}

pub struct DataLoader<T: Primitive, U: Primitive> {
    // dataset should be copiable (like the torch one that's copied and sharded over processes)
    dataset: Box<dyn Dataset<T, U>>,
    batch_size: usize,
    shuffle: bool,
    curr_index: usize,
    indices: Vec<usize>,
    drop_last: bool,
    seed: u64,
    rng: StdRng,
}

impl<T: Primitive, U: Primitive> DataLoader<T, U> {
    pub fn new(dataset: Box<dyn Dataset<T, U>>, seed: u64) {
        let indices: Vec<_> = (0..dataset.len()).collect();
        let rng = StdRng::seed_from_u64(seed);
    }
}

// TODO put in threads
impl<T: Primitive, U: Primitive> Iterator for DataLoader<T, U> {
    type Item = (Tensor<T>, Tensor<U>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_index < self.dataset.len() {
            let batch_indices = &self.indices[self.curr_index..self.curr_index + self.batch_size];

            // prepare data for stacking
            let mut data = Vec::new();
            let mut targets = Vec::new();
            for &i in batch_indices {
                let (x, y) = self.dataset.get_item(i);
                data.push(x);
                targets.push(y);
            }
            // stack data to create batch dim
            // ndarray::stack(Axis(0), &data[..]);

            // shuffling indices for next epoch when we passed through the whole dataset
            if self.curr_index + self.batch_size >= self.dataset.len() {
                self.indices.shuffle(&mut self.rng);
            }

            self.curr_index += self.batch_size;
            Some(Ok((data, targets)))
        } else {
            Some(Err("Index out of bounds!"))
        }
    }
}
