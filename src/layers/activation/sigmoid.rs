//! Applies the nonlinear Log-Sigmoid function.
//!
//! Non-linearity activation function: y = (1 + e^(-x))^(-1)
//!
//! A classic choice in neural networks.
//! But you might consider using ReLu as an alternative.
//!
//! ReLu, compared to Sigmoid
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this
//! writing (2015).
use co::backend::IBackend;
use co::tensor::SharedTensor;
use conn::plugin::INn;
use layer::*;
use shared_memory::*;
use std::sync::{Arc, RwLock};

#[derive(Debug, Copy, Clone)]
/// Sigmoid Activation Layer
pub struct Sigmoid;

impl<B: IBackend + INn<f32>> ILayer<B> for Sigmoid {
    impl_ilayer_activation!();

    fn reshape(&mut self, bottom: &[ArcLock<HeapBlob>], top: &mut Vec<ArcLock<HeapBlob>>) {
        let btm = bottom[0].read().unwrap();
        top[0] = Arc::new(RwLock::new(Blob::from_data(SharedTensor::<f32>::new(btm.data().latest_device(), btm.shape()).unwrap())));
    }

    fn forward_layer(&self, backend: &B, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>) {
        let bottom_data = bottom[0].data();
        let top_data = top[0].mut_data();

        backend.sigmoid_plain(bottom_data, top_data).unwrap();
        debug!("SIGMOID HERE");
    }

    fn backward_layer(&self, backend: &B, top: &[ReadBlob], propagate_down: &[bool], bottom: &mut Vec<&mut WriteBlob>) {
        if propagate_down[0] {
            let _ = backend.sigmoid_grad_plain(top[0].data(),
                                               top[0].diff(),
                                               top[0].data(),
                                               bottom[0].mut_diff());
        }
    }
}

impl Sigmoid {
    fn sigmoid(z: f32) -> f32 {
        1f32 / (1f32 + (-z).exp())
    }

    fn sigmoid_prime(z: f32) -> f32 {
        Sigmoid::sigmoid_prime_precalc(Sigmoid::sigmoid(z))
    }

    fn sigmoid_prime_precalc(sigmoid_z: f32) -> f32 {
        sigmoid_z * (1f32 - sigmoid_z)
    }
}
