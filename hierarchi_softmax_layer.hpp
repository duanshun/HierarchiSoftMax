#ifndef CAFFE_HIERARCHI_SOFTMAX_LAYER_HPP_
#define CAFFE_HIERARCHI_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class HierarchiSoftMaxLayer : public NeuronLayer<Dtype> {
public:
    explicit HierarchiSoftMaxLayer(const LayerParameter& param): NeuronLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "HierarchiSoftMax"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    int cat_id;
    int nSample;
    int nClass;
};


}  // namespace caffe

#endif  // CAFFE_HIERARCHI_SOFTMAX_LAYER_HPP_