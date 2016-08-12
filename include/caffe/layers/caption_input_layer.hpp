#ifndef CAFFE_CAPTION_INPUT_LAYER_HPP_
#define CAFFE_CAPTION_INPUT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CaptionInputLayer : public Layer<Dtype> {
 public:
  explicit CaptionInputLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CaptionInput"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int N_; // batch size
  int V_; // vocabulary size
  int P_; // Img Size
  
  Dtype clipping_threshold_; // threshold for clipped gradient
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> be_multiplier_;

  Blob<Dtype> X_;   // img_vector x Wi ; word_vectors x Ws

};

}  // namespace caffe

#endif  // CAFFE_CAPTION_INPUT_LAYER_HPP_
