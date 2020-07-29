use crate::prelude::{matmul, matsum};
use rand::{
    distributions::{Standard, Uniform},
    Rng,
};
use std::borrow::Borrow;
use std::ops::{Add, Mul};
#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub dim: [usize; 2],
}

#[allow(unused_variables)]
impl Tensor {
    pub fn new(data: &[f32], dim: &[usize]) -> Self {
        todo!()
    }
    pub fn randn(dim: &[usize]) -> Self {
        let rng = rand::thread_rng();
        let data_len = dim.iter().fold(1, |acc, x| acc * x);
        let data: Vec<f32> = rng.sample_iter(Standard).take(data_len).collect();
        let dim = slice_to_dim(dim);
        Tensor { data, dim }
    }
    pub fn randu(dim: &[usize]) -> Self {
        let between = Uniform::from(0.0..1.0);
        let rng = rand::thread_rng();
        let data_len = dim.iter().fold(1, |acc, x| acc * x);
        let data: Vec<f32> = rng.sample_iter(between).take(data_len).collect();
        let dim = slice_to_dim(dim);
        Tensor { data, dim }
    }
    pub fn zeros(dim: &[usize]) -> Self {
        let data_len = dim.iter().fold(1, |acc, x| acc * x);
        let data: Vec<f32> = vec![0.0; data_len];
        let dim = slice_to_dim(dim);
        Tensor { data, dim }
    }
    pub fn ones(dim: &[usize]) -> Self {
        todo!()
    }
    pub fn uniform_(&self, lo: f32, up: f32) -> Self {
        todo!()
    }
    pub fn fill_(&self, v: f32) -> Self {
        let data = self.data.iter().map(|_| v).collect();
        Tensor {
            data,
            dim: self.dim,
        }
    }
    pub fn copy_(&mut self, src: &Tensor) -> Self {
        todo!()
    }
    pub fn size(&self) -> [usize; 2] {
        let out = self.dim;
        out
    }
    pub fn randn_like(&self) -> Self {
        todo!()
    }
    pub fn shallow_clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            dim: self.dim.clone(),
        }
    }

    pub fn tr(&self) -> Self {
        todo!()
    }
    // NN
    pub fn conv1d<T: Borrow<Tensor>>(
        &self,
        weight: &Tensor,
        bias: Option<T>,
        stride: &[u64],
        padding: &[u64],
        dilation: &[u64],
        groups: u64,
    ) -> Tensor {
        todo!()
    }

    pub fn conv2d<T: Borrow<Tensor>>(
        &self,
        weight: &Tensor,
        bias: Option<T>,
        stride: &[u64],
        padding: &[u64],
        dilation: &[u64],
        groups: u64,
    ) -> Tensor {
        todo!()
    }

    pub fn conv3d<T: Borrow<Tensor>>(
        &self,
        weight: &Tensor,
        bias: Option<T>,
        stride: &[u64],
        padding: &[u64],
        dilation: &[u64],
        groups: u64,
    ) -> Tensor {
        todo!()
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let data = matmul(&self.data, &rhs.data, self.dim[0], rhs.dim[1], self.dim[1]);
        let dim = [self.dim[0], rhs.dim[1]];
        Tensor { data, dim }
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let data = matsum(&self.data, &rhs.data);
        let dim = [rhs.dim[1], self.dim[0]];
        Tensor { data, dim }
    }
}

fn slice_to_dim(s: &[usize]) -> [usize; 2] {
    let mut d = [0usize; 2];
    for (i, e) in s.iter().enumerate().take(2) {
        d[i] = *e;
    }
    d
}
