#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/hierarchi_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HierarchiSoftMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_gpu_data();

    const Dtype* cur_prob = bottom[0]->gpu_data();
    const Dtype* pre_prob = bottom[1]->gpu_data();

    int id = 0;
    int C1 = bottom[1]->shape()[1];
    for (int i = 0, k = 0; i < nSample; ++i){
        id = i*C1 + cat_id;
        for (int j = 0; j < nClass; j++, k++)
            top_data[k] = cur_prob[k]*pre_prob[id];
    }
}

template <typename Dtype>
void HierarchiSoftMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data0 = bottom[0]->gpu_data();
    const Dtype* bottom_data1 = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    int C1 = bottom[1]->shape()[1];
    if (propagate_down[0]) {
        int id = 0;
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        for (int i = 0, k=0; i < nSample; ++i) {
            id = i*C1 + cat_id;
            for (int j = 0; j < nClass; j++, k++)
                bottom_diff[k] = top_diff[k]*bottom_data1[id];
         }
    }

    if (propagate_down[1]){
        Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
        const int M = bottom[1]->count();
        caffe_set(M, Dtype(0), bottom_diff);
        float sum = 0;

        for (int i = 0, k = 0; i < nSample; ++i){
            sum = 0;
            for (int j = 0; j < nClass; j++, k++)
                sum += bottom_data0[k]*top_diff[k];
            bottom_diff[i*C1+cat_id] = sum;
        }

    }
}

INSTANTIATE_LAYER_GPU_FUNCS(HierarchiSoftMaxLayer);

}  // namespace caffe
