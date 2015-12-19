//! Provides methods to calculate the loss (cost) of some output.
//!
//! A loss function is also sometimes called cost function.
#[macro_export]
macro_rules! impl_ilayer_loss {
    () => (
        fn exact_num_top_blobs(&self) -> usize { 1 }
        fn exact_num_bottom_blobs(&self) -> usize { 1 }
        fn auto_top_blobs(&self) -> bool { true }
    )
}

pub use self::softmax::SoftmaxLoss;

pub mod softmax;
