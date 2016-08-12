#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/caption_input_layer.hpp"

namespace caffe {


template <typename Dtype>
void CaptionInputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  /*
  bottom[0]:   img vector
  bottom[1]:   sentence words
  */
  
  N_ = bottom[0]->num();
  H_ = this->layer_param_.caption_input_param().hidden_size(); // number of hidden units
  P_ = this->layer_param_.caption_input_param().image_size();
  V_ = this->layer_param_.caption_input_param().vocabulary_size();
  T_ = this->layer_param_.caption_input_param().sequence_len() + 1;

  CHECK_EQ(bottom[1]->count()/bottom[1]->num() + 1, T_);   // img + sentence length
  CHECK_EQ(bottom[0]->count()%bottom[0]->num(), 0);
  CHECK_EQ(bottom[1]->count()%bottom[1]->num(), 0);
  CHECK_EQ(bottom[0]->count()/bottom[0]->num(), P_);  // check image vector dimension
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());     // check batch size should be equal for img and sents

  LOG(INFO) << "===========  Caption Input Setup ============"<<std::endl;
  LOG(INFO) << "~~~~~ Bottom Size "<<bottom.size()<<std::endl;
  LOG(INFO) << "~~~~~ Top Size "<<top.size()<<std::endl;
  LOG(INFO) << "~~~~~ N(batch) "<<N_<<std::endl;
  LOG(INFO) << "~~~~~ H(hidden) "<<H_<<std::endl;
  LOG(INFO) << "~~~~~ P(imgVec) "<<P_<<std::endl;
  LOG(INFO) << "~~~~~ V(voc) "<<V_<<std::endl;
  LOG(INFO) << "~~~~~ T "<<T_<<std::endl;

  LOG(INFO) << "Bottom 0 size "<< bottom.size()<<"  "<<bottom[0]->count()<<" "<<bottom[0]->num()<<std::endl;
  LOG(INFO) << "Bottom 1 size "<< bottom.size()<<"  "<<bottom[1]->count()<<" "<<bottom[1]->num()<<std::endl;
  LOG(INFO) << "Bottom 0 size "<< bottom.size()<<"  "<<bottom[0]->shape(0)<<" "<<bottom[0]->shape(1)<<std::endl;
  LOG(INFO) << "Bottom 1 size "<< bottom.size()<<"  "<<bottom[1]->shape(0)<<" "<<bottom[1]->shape(1)<<std::endl;

  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } 
  else 
  {
	 
	// 0 is We, 1 is be -- for image encoding
	// 2 is Ws, for word encoding
    this->blobs_.resize(3);
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.caption_input_param().weight_filler()));
	  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.caption_input_param().bias_filler()));


    vector<int> weight_shape;

    // We: img_size x hidden_size
    weight_shape.clear();
    weight_shape.push_back(H_);
    weight_shape.push_back(P_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // be: 1xhidden_size
    vector<int> bias_shape(1, H_);
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    bias_filler->Fill(this->blobs_[1].get());
    
    // Ws: voc_size x hidden_size
    weight_shape.clear();
    weight_shape.push_back(V_);
    weight_shape.push_back(H_);
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[2].get());


  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  
  for(int i=0; i<this->blobs_.size(); i++)
  {
    LOG(INFO) << "Caption Input Blobs "<<i<<" shape: "<<this->blobs_[i].get()->shape_string() <<std::endl;
  }
  


}

/*
top size: top[0]: (T*N, H)  320x50
Hout_ size:  (T, N, H)      320x1x50
*/
template <typename Dtype>
void CaptionInputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  //LOG(INFO) << "~~~~~ T(max sent lengths) "<<T_<<std::endl;
  
  vector<int> original_top_shape;
  original_top_shape.push_back(T_*N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);


  //stack image and word vector
  vector<int> X_shape; 
  X_shape.push_back(T_);
  X_shape.push_back(N_);
  X_shape.push_back(H_);
  X_.Reshape(X_shape);
  X_.ShareData(*top[0]);
  X_.ShareDiff(*top[0]);
  
  
  //  Set up the be multiplier  -------------  X = We * img_vec + be
  vector<int> be_multiplier_shape(1, 1*N_);
  be_multiplier_.Reshape(be_multiplier_shape);
  caffe_set(be_multiplier_.count(), Dtype(1), be_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void CaptionInputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  const Dtype* bottom_img_data = bottom[0]->cpu_data();
  const Dtype* bottom_sent_data = bottom[1]->cpu_data();
  
  const Dtype* We = this->blobs_[0]->cpu_data();
  const Dtype* be = this->blobs_[1]->cpu_data();
  const Dtype* Ws = this->blobs_[2]->cpu_data();


  Dtype* X_data = X_.mutable_cpu_data();


  // Compute X, two steps:  X = We * img_vec, X += outerprod(1, be)
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1*N_, H_, P_, Dtype(1.), bottom_img_data, We, Dtype(0.), X_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1*N_, H_, 1, Dtype(1.), be_multiplier_.cpu_data(), be, Dtype(1.), X_data);

  
  for(int t=1; t < T_; t++)
  {
    for(int n=0; n< N_; n++)
    {
      int word_id = int(*(bottom_sent_data + bottom[1]->offset(n, t-1)));
      CHECK_EQ(n*(T_-1) + t-1, bottom[1]->offset(n, t-1));

      if(word_id>=0)
      {
        caffe_copy(H_, Ws + this->blobs_[2]->offset(word_id), X_data + X_.offset(t,n));
      } 
    }
  }
  

}

template <typename Dtype>
void CaptionInputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  //LOG(INFO) << "===========  Backward ============"<<std::endl;
    
  const Dtype* bottom_img_data = bottom[0]->cpu_data();
  const Dtype* bottom_sent_data = bottom[1]->cpu_data();
  
  const Dtype* dX = X_.cpu_diff();  
  Dtype* dWe = this->blobs_[0]->mutable_cpu_diff();
  Dtype* dbe = this->blobs_[1]->mutable_cpu_diff();
  Dtype* dWs = this->blobs_[2]->mutable_cpu_diff();
  
  
  
    
  // X[0] = X_img * We + be * be_multiplier
  // dWe = dX[0]' * X_img
  // HxP = (NxH)' * (NxP)
  if (this->param_propagate_down_[0]) 
  {
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, H_, P_, N_, Dtype(1.), dX, bottom_img_data, Dtype(1.), dWe);
  }
  
  
  // dbe
  if (this->param_propagate_down_[1]) 
  {
  
    //caffe_cpu_gemm(CblasTrans, CblasNoTrans, H_, Dtype(1.), N_, Dtype(1.), dX, be_multiplier_.cpu_data(), Dtype(0.), dbe);
    caffe_cpu_gemv(CblasTrans, N_, H_, Dtype(1.), dX, be_multiplier_.cpu_data(), Dtype(1.), dbe);
  }
  
  
  if (this->param_propagate_down_[2]) 
  {
    //dWs
    for(int t=1; t < T_; t++)
    {
      for(int n=0; n< N_; n++)
      {
        int word_id = int(*(bottom_sent_data + bottom[1]->offset(n, t-1)));
        caffe_axpy(H_, Dtype(1.), dX + X_.offset(t,n),  dWs + this->blobs_[2]->offset(word_id)); 
      }
    }
  }
  
  
  //LOG(INFO) << "===========  Backward Complete============"<<std::endl;

}

#ifdef CPU_ONLY
STUB_GPU(CaptionInputLayer);
#endif

INSTANTIATE_CLASS(CaptionInputLayer);
REGISTER_LAYER_CLASS(CaptionInput);

}  // namespace caffe
