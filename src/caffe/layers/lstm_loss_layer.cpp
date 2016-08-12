#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/lstm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LstmLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  //LossLayer<Dtype>::LayerSetUp(bottom, top);  //write out explicitly from loss_layer.cpp
  if (this->layer_param_.loss_weight_size() == 0) {
      this->layer_param_.add_loss_weight(Dtype(1));
  }
  
  LOG(INFO) << "============ LstmLossLayer ============   "<<std::endl;

  N_ = bottom[1]->shape(0);
  D_ = bottom[0]->shape(1);
  T_ = bottom[0]->shape(0) / N_ ;

  LOG(INFO) << "Top size "<<top.size()<<std::endl;
  LOG(INFO) << "N "<<N_<<std::endl;
  LOG(INFO) << "D "<<D_<<std::endl;
  LOG(INFO) << "T "<<T_<<std::endl;

  CHECK_EQ(bottom[0]->count(0) % N_ , 0);
  CHECK_EQ(T_ , bottom[1]->shape(1));


  //create internel softmax layer
  //add value in prob_
  
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);  
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);


  // ignore "-1" label as blank
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  
  
  if(top.size()==2)
  {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void LstmLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  //LossLayer<Dtype>::Reshape(bottom, top);  //to avoid ChECK FAIL, write out explicitly from loss_layer.cpp
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  if (top.size() >= 2) 
  {
    top[1]->ReshapeLike(*bottom[0]);
  }

}


template <typename Dtype>
Dtype LstmLossLayer<Dtype>::get_normalizer(LossParameter_NormalizationMode normalization_mode, int valid_count) 
{
  return N_;
}


template <typename Dtype>
void LstmLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  //LOG(INFO) << "===========  LstmLoss Forward ============"<<std::endl;
  //LOG(INFO) << *(bottom[0]->cpu_data())<<" "<<*(bottom[0]->cpu_data()+10);

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();  // in our case, N x T

  
  Dtype loss = 0;
  for (int t = 0; t <T_ ; t++) 
  {
    // see GPU LstmLossForwardGPU
    for (int n = 0; n < N_; n++) 
    {
      const int label_value = static_cast<int>(label[n * T_ + t]);
      //LOG(INFO) << t << " "<<label_value;
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
  
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
        
      // check OK, sum to 1
      loss -= log(std::max(prob_data[ (t * N_ + n) * D_ + label_value], Dtype(FLT_MIN)));
    }
  }
  
  //normalize by batch size
  //top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  top[0]->mutable_cpu_data()[0] = loss / N_;
  
  
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }

}

template <typename Dtype>
void LstmLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) 
  {
    //LOG(INFO) << "===========  LstmLoss Backward ============"<<std::endl;

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    

    for (int t = 0; t < T_; t++) 
    {
      for (int n = 0; n < N_; n++) 
      {
        const int label_value = static_cast<int>(label[n * T_ + t]);
                
        if (has_ignore_label_ && label_value == ignore_label_) 
        {
          for (int d = 0; d < D_; ++d) {
            bottom_diff[(t * N_ + n) * D_ + d] = 0;
          } 
        } 
        else {
          //CHECK_LT(label_value, D_);
          //CHECK_GE(label_value, 0);
          bottom_diff[ (t * N_ + n) * D_ + label_value ] -= 1;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LstmLossLayer);
#endif

INSTANTIATE_CLASS(LstmLossLayer);
REGISTER_LAYER_CLASS(LstmLoss);

}  // namespace caffe
