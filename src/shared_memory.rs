//!
//!
//! This is quite unimportant and might be refactored soon.
//!
//! See [Issue #22][issue] for more informations.
//! [issue]: https://github.com/autumnai/leaf/issues/22
use std::sync::{Arc, RwLock};
use coblas::plugin::IBlas;
use co::backend::IBackend;
use co::tensor::*;
use co::plugin::numeric_helpers::cast;

/// shared Lock used for our memory blobs
pub type ArcLock<T> = Arc<RwLock<T>>;
/// Blob allocated on the heap via SharedTensor
pub type HeapBlob = Blob<f32>;
/// SharedTensor with f32 values
pub type HeapTensor = SharedTensor<f32>;

#[derive(Debug)]
/// TODO
pub struct Blob<T> {
    data: SharedTensor<T>,
    diff: SharedTensor<T>
}

impl<T> Blob<T> {
    /// Create a Blob by creating a fitting diff for the data.
    ///
    /// This allocates a new SharedTensor of the same size as the provided `data`.
    /// The new SharedTensor will act as the diff to the data.
    pub fn from_data(data: SharedTensor<T>) -> Blob<T> {
        let diff = SharedTensor::new(data.latest_device(), data.desc()).unwrap();
        Blob {
            data: data,
            diff: diff
        }
    }

    /// Returns a String representation of the Blobs' `shape`
    ///
    /// The first numbers represent the size of the dimension.
    /// The last number in brackets defines the dimensionality of the Blob.
    pub fn shape_string(&self) -> String {
        let mut string: String = "".to_owned();
        for dim in self.data().desc().dims().clone() {
            string.push_str(&format!("{} ", &dim.to_string()));
        }
        string.push_str(&format!("({})", self.data().desc().rank().to_string()));
        string
    }

    /// Returns the shape of the Blob.
    pub fn shape(&self) -> &Vec<usize> {
        self.data().desc().dims()
    }

    /// Returns the numer of elements in each of `data` and `diff`.
    pub fn size(&self) -> usize {
        self.data().desc().size()
    }

    /// Returns a reference to the data of the Blob.
    pub fn data(&self) -> &SharedTensor<T> {
        &self.data
    }

    /// Returns a mutable reference to the data of the Blob.
    pub fn mut_data(&mut self) -> &mut SharedTensor<T> {
        &mut self.data
    }

    /// Returns a reference to the diff of the Blob.
    pub fn diff(&self) -> &SharedTensor<T> {
        &self.diff
    }

    /// Returns a mutable reference to the diff of the Blob.
    pub fn mut_diff(&mut self) -> &mut SharedTensor<T> {
        &mut self.diff
    }
}

impl Blob<f32> {
    /// Apply the diff to the data.
    ///
    /// In machine learnig this is used when the blob represents weights in a network,
    /// and the computed gradients should be applied to weights.
    #[allow(unused_must_use)]
    pub fn apply_diff<B: IBlas<f32> + IBackend>(&mut self, backend: &B) {
        let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &1).unwrap();
        let local_a = [cast::<f32, f32>(-1f32).unwrap()];
        {
            let a = shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap();
            a.as_mut_slice().clone_from_slice(&local_a);
        }
        backend.axpy_plain(shared_a, &self.diff, &mut self.data);
    }

    /// Calculate the sum over the squares of each diff element and store it in `res`.
    #[allow(unused_must_use)]
    pub fn sumsq_diff<B: IBlas<f32>>(&self, backend: &B, res: &mut SharedTensor<f32>) {
        backend.dot_plain(&self.diff, &self.diff, res);
    }

    /// TODO
    pub fn regularize_l2<B: IBlas<f32>>(&mut self, backend: &B, decay: &SharedTensor<f32>) {
        let _ = backend.axpy_plain(decay, &self.data, &mut self.diff);
    }
}
