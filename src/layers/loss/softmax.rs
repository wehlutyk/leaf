//! Computes the multinomial logistic loss of the softmax of its bottom Blob.
//!
//! This is conceptually identical to a softmax layer followed by a multinomial
//! logistic loss layer, but provides a more numerically stable gradient.

use co::backend::IBackend;
use co::tensor::{SharedTensor, ITensorDesc};
use conn::plugin::INn;
use layer::*;
use util::native_backend;
use std::rc::Rc;
use std::f32;
use co::plugin::numeric_helpers::*;

#[derive(Debug)]
/// Softmax Loss Layer
pub struct SoftmaxLoss {
    softmax_axis: Option<usize>,
    // outer_num: Option<usize>,
    // inner_num: Option<usize>,

    label_copy: Option<SharedTensor<f32>>,
}

impl<B: IBackend + INn<f32>> ILayer<B> for SoftmaxLoss {
    impl_ilayer_loss!();

    fn init(&mut self, backend: &Rc<B>) {
        self.softmax_axis = Some(1); // TODO: make configurable
        // let bottom: ReadBlob = unimplemented!();
        // let outer_num = bottom.shape().iter().take(softmax_axis + 1).fold(1, |prod, i| prod * i);
        // let inner_num = bottom.shape().iter().skip(softmax_axis + 1).fold(1, |prod, i| prod * i);
        self.reshape();
    }

    fn forward_layer(&self, backend: &B, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>) {
        let bottom_data = bottom[0].data();
        let label = bottom[1].data();
        let probability = top[0].mut_data();

        let _ = backend.softmax_plain(bottom_data, probability);

        let native = native_backend();
        // match label.add_device(native.device()) { _ => label.sync(native.device()).unwrap() }
        let native_label = label.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();
        match probability.add_device(native.device()) { _ => probability.sync(native.device()).unwrap() }
        let native_probability = probability.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();

        let mut loss = 0f32;
        let outer_num = bottom_data.desc().iter().take(self.softmax_axis.unwrap() + 1).fold(1, |prod, i| prod * i);
        let inner_num = bottom_data.desc().iter().skip(self.softmax_axis.unwrap() + 1).fold(1, |prod, i| prod * i);
        let dim: usize = probability.desc().size() / outer_num;
        for i in 0..(outer_num - 1) {
            for j in 0..(inner_num - 1) {
                let label_value: usize = native_label[i * inner_num + j] as usize;
                // if (has_ignore_label_ && label_value == ignore_label_) {
                //     continue;
                // }
                // DCHECK_GE(label_value, 0);
                // DCHECK_LT(label_value, prob_.shape(softmax_axis_));
                loss -= native_probability[i * dim + label_value * inner_num + j].max(f32::MIN).ln();
            }
        }
        debug!("SOFTMAX HERE");
        // Dtype loss = 0;
        // for (int i = 0; i < outer_num_; ++i) {
        //     for (int j = 0; j < inner_num_; j++) {
        //         const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        //         if (has_ignore_label_ && label_value == ignore_label_) {
        //             continue;
        //         }
        //         DCHECK_GE(label_value, 0);
        //         DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        //         loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)));
        //         ++count;
        //     }
        // }
    }

    fn backward_layer(&self, backend: &B, top: &[ReadBlob], propagate_down: &[bool], bottom: &mut Vec<&mut WriteBlob>) {
        if propagate_down[0] {
            let _ = backend.softmax_grad_plain(top[0].data(),
                                               top[0].diff(),
                                               bottom[0].mut_diff());
        }
    }

}

impl SoftmaxLoss {
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

impl ::std::default::Default for SoftmaxLoss {
    fn default() -> SoftmaxLoss {
        SoftmaxLoss {
            softmax_axis: None,
            label_copy: None,
        }
    }
}
