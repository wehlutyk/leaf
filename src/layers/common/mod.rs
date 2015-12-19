//! Provides common neural network layers.
//!
//! For now the layers in common should be discribed as layers that are typical
//! layers for building neural networks but are not activation or loss layers.
#[macro_export]
macro_rules! impl_ilayer_common {
    () => (
        fn exact_num_top_blobs(&self) -> usize { 1 }
        fn exact_num_bottom_blobs(&self) -> usize { 1 }
    )
}

pub use self::convolution::Convolution;
pub use self::softmax::Softmax;

pub mod convolution;
pub mod softmax;
