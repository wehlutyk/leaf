//! TODO
use co::backend::{Backend, IBackend, BackendConfig};
use co::framework::IFramework;
use co::frameworks::Native;
use co::tensor::SharedTensor;
use co::plugin::numeric_helpers::*;
use coblas::plugin::IBlas;

/// Create a simple native backend
///
/// This is handy when you need to sync data to host memory to read/write it.
pub fn native_backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = framework.hardwares();
    let backend_config = BackendConfig::new(framework, hardwares);
    Backend::new(backend_config).unwrap()
}

/// Extends IBlas with Axpby
pub trait IBlasAxpby<F: Float> : IBlas<F> {
    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby_plain(&self, a: &SharedTensor<F>, x: &SharedTensor<F>, b: &SharedTensor<F>, y: &mut SharedTensor<F>) -> Result<(), ::co::error::Error> {
        try!(self.scal_plain(b, y));
        try!(self.axpy_plain(a, x, y));
        Ok(())
    }
}

impl<T: IBlas<f32>> IBlasAxpby<f32> for T {}
