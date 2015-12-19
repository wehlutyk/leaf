//! Applies Softmax to the top Blob

use layer::*;
use co::backend::IBackend;
use conn::plugin::INn;
use std::rc::Rc;

#[derive(Debug, Copy, Clone)]
/// Softmax Layer
pub struct Softmax;

impl<B: IBackend + INn<f32>> ILayer<B> for Softmax {
    impl_ilayer_common!();

    fn init(&mut self, backend: &Rc<B>) {
        self.reshape();
    }

    fn forward_layer(&self, backend: &B, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>) {
        let bottom_data = bottom[0].data();
        let top_data = top[0].mut_data();

        let _ = backend.softmax_plain(bottom_data, top_data);
    }

    fn backward_layer(&self, backend: &B, top: &[ReadBlob], propagate_down: &[bool], bottom: &mut Vec<&mut WriteBlob>) {
        if propagate_down[0] {
            let _ = backend.softmax_grad_plain(top[0].data(),
                                               top[0].diff(),
                                               bottom[0].mut_diff());
        }
    }
}

impl Softmax {
    fn reshape(&mut self) {
        // let softmax_axis = 1; // TODO: make configurable
        // let bottom: ReadBlob = unimplemented!();
        // let outer_num = bottom.shape().iter().take(softmax_axis + 1).fold(1, |prod, i| prod * i);
        // let inner_num = bottom.shape().iter().skip(softmax_axis + 1).fold(1, |prod, i| prod * i);

        // softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
        // top[0]->ReshapeLike(*bottom[0]);
        // vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
        // sum_multiplier_.Reshape(mult_dims);
        // Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
        // caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
        // outer_num_ = bottom[0]->count(0, softmax_axis_);
        // inner_num_ = bottom[0]->count(softmax_axis_ + 1);
        // vector<int> scale_dims = bottom[0]->shape();
        // scale_dims[softmax_axis_] = 1;
        // scale_.Reshape(scale_dims);
    }
}
