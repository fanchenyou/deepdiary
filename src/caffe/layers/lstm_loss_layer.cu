#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/lstm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LstmLossForwardGPU(const int nthreads, 
          const int N_, const int T_, const int D_,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_ ) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int t = index / N_;
    const int n = index % N_;
    const int ix = n * T_ + t; //bottom[1] dim: NxT
    
    const int label_value = static_cast<int>(label[ix]);
    
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      loss[ix] = 0;
    } 
    else 
    {
      //printf("%d %d %d %f\n", n, t, label_value, prob_data[ index * D_ + label_value]);
      loss[ix] = -log(max(prob_data[ index * D_ + label_value], Dtype(FLT_MIN)));
    }
  }
}

template <typename Dtype>
__global__ void ShowPred(const int nthreads, 
          const int N_, const int T_, const int D_,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_ ) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //for(int n=0; n<N_; n++)
    int n = 1;
    {
    printf("Index %d===\n", n);
    printf("GT \n");
    const int ix = n * T_ ; //bottom[1] dim: NxT
    int max_len = 0;
    for(int t=0; t<T_; t++)
    {
      int lb = int(label[ix+t]);
      if(lb==ignore_label_)
      {
        break;
      }
      max_len += 1;
      printf("%d ", lb);
    }
    printf("\n");
    printf("Pred \n");
    for(int t=0; t<max_len; t++)
    {
      int max_i = 0;
      Dtype max_prob = 0.0;
      for(int v=0; v<D_;v++)
      {
        Dtype tmp = prob_data[t*N_*D_ + n*D_+v];
        if(max_prob<tmp)
        {
          max_prob = tmp;
          max_i = v;
        }
      }
      printf("%d ", max_i);

    }

      printf("----\n");
    }
  }
}

      
      

template <typename Dtype>
void LstmLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  //LOG(INFO) << "===========  LstmLoss Forward GPU ============"<<std::endl;

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = T_ * N_;
  

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  // So there are T_ x N_ individual losses for each word, which is of same size with bottom[1]
  Dtype* loss_data = bottom[1]->mutable_gpu_diff();
  caffe_gpu_set(bottom[1]->count(), Dtype(0.), loss_data);

  LstmLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
            (nthreads, N_, T_, D_, prob_data, label, loss_data, has_ignore_label_, ignore_label_);
  
  /*
  ShowPred<Dtype><<<CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS>>>
            (1, N_, T_, D_, prob_data, label, loss_data, has_ignore_label_, ignore_label_);
  */     
            
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);

  top[0]->mutable_cpu_data()[0] = loss / N_;
  
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}


          
template <typename Dtype>
__global__ void LstmLossBackwardGPU(const int nthreads, 
          const int N_, const int T_, const int D_,
          const Dtype* label, Dtype* bottom_diff, 
          const bool has_ignore_label_, const int ignore_label_ ) 
{

  CUDA_KERNEL_LOOP(index, nthreads) 
  {
  
    const int t = index / N_;
    const int n = index % N_;
    const int ix = n * T_ + t; //bottom[1] dim: NxT
    const int label_value = static_cast<int>(label[ix]);
    const int p = index * D_;

    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      for (int d = 0; d < D_; ++d) {
        bottom_diff[p + d] = 0;
      }
    } 
    else 
    {
      bottom_diff[p + label_value] -= 1;
    }
  }
}

template <typename Dtype>
void LstmLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
  
    //LOG(INFO) << "===========  LstmLoss GPU Backward ============"<<std::endl;

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = T_ * N_;
    CHECK_EQ(prob_.count(), T_*N_*D_);
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    

    LstmLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, T_, D_, label, bottom_diff,
        has_ignore_label_, ignore_label_);

    const Dtype loss_weight = top[0]->cpu_diff()[0] / N_;
                              
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LstmLossLayer);

}  // namespace caffe
