#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/caption_input_layer.hpp"

namespace caffe {




template <typename Dtype>
__global__ void GetGT(const int nthreads, const int N_, const int T_, const int H_, 
                      int t, const Dtype* gt, const Dtype* Ws, Dtype* X_data) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / H_;
    const int h = index % H_;
    const Dtype *lb = gt + n*(T_-1) ;
    const int word_id= int(lb[t-1]);

    const Dtype *Ws_id = Ws + word_id*H_ ;
    Dtype *X_t = X_data + t*N_*H_ + n*H_;
    X_t[h] = Ws_id[h];
    
  }
}


template <typename Dtype>
__global__ void UpdateWs(const int nthreads, const int N_, const int T_, const int H_, 
                      int t,  const Dtype* gt, const Dtype* dX, Dtype* dWs) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / H_;
    const int h = index % H_;
    const Dtype *lb = gt + n*(T_-1) ;
    const int word_id= int(lb[t-1]);

    Dtype *dWs_id = dWs + word_id*H_ ;
    const Dtype *dX_t = dX + t*N_*H_ + n*H_;
    dWs_id[h] += dX_t[h];
    
  }
}




template <typename Dtype>
void CaptionInputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(top[0]->gpu_data(), X_.gpu_data());


  const Dtype* bottom_img_data = bottom[0]->gpu_data();
  const Dtype* bottom_sent_data = bottom[1]->gpu_data();
  
  const Dtype* We = this->blobs_[0]->gpu_data();
  const Dtype* be = this->blobs_[1]->gpu_data();
  const Dtype* Ws = this->blobs_[2]->gpu_data();

  Dtype* X_data = X_.mutable_gpu_data();




  // Compute X[0], two steps:  X[0] = We * img_vec, X[0] += outerprod(1, be)
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1*N_, H_, P_, Dtype(1.), bottom_img_data, We, Dtype(0.), X_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1*N_, H_, 1, Dtype(1.), be_multiplier_.gpu_data(), be, Dtype(1.), X_data);


  for (int t = 1; t < T_; t++) 
  {
      GetGT<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, N_, T_, H_, t, bottom[1]->gpu_data(), Ws, X_data);
      CUDA_POST_KERNEL_CHECK;
  }
  
}

template <typename Dtype>
void CaptionInputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{

  //LOG(INFO) << "===========  Backward Caption Input GPU ============"<<std::endl;

  const Dtype* bottom_img_data = bottom[0]->gpu_data();
  const Dtype* bottom_sent_data = bottom[1]->gpu_data();
  

  const Dtype* dX = X_.gpu_diff();
  Dtype* dWe = this->blobs_[0]->mutable_gpu_diff();
  Dtype* dbe = this->blobs_[1]->mutable_gpu_diff();
  Dtype* dWs = this->blobs_[2]->mutable_gpu_diff();
 
     
  if (this->param_propagate_down_[0]) 
  {
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, H_, P_, N_, Dtype(1.), 
                    dX, bottom_img_data, Dtype(1.), dWe);
  }

  if (this->param_propagate_down_[1]) 
  { 
    caffe_gpu_gemv(CblasTrans, N_, H_, Dtype(1.), 
                    dX, be_multiplier_.gpu_data(), Dtype(1.), dbe);
  }
  
  if (this->param_propagate_down_[2]) 
  {
    for(int t=1; t < T_; t++)
    {
      UpdateWs<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
          N_*H_, N_, T_, H_, t, bottom[1]->gpu_data(), dX, dWs);
      CUDA_POST_KERNEL_CHECK;
    } 
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(CaptionInputLayer);

}  // namespace caffe
