#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void LstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  /*
  bottom[0]:   img vector + word vectors
  */
  
  clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
  H_ = this->layer_param_.lstm_param().hidden_size(); // number of hidden units
  V_ = this->layer_param_.lstm_param().vocabulary_size();
  T_ = this->layer_param_.lstm_param().sequence_len() + 1;
  N_ = bottom[0]->num() / T_;  //input is T*N

  
  LOG(INFO) << "===========  LSTM Setup ============"<<std::endl;
  LOG(INFO) << "~~~~~ Bottom Size "<<bottom.size()<<std::endl;
  LOG(INFO) << "~~~~~ Top Size "<<top.size()<<std::endl;
  LOG(INFO) << "~~~~~ N(batch) "<<N_<<std::endl;
  LOG(INFO) << "~~~~~ H(hidden) "<<H_<<std::endl;
  LOG(INFO) << "~~~~~ V(voc) "<<V_<<std::endl;
  LOG(INFO) << "~~~~~ T "<<T_<<std::endl;

  LOG(INFO) << "Bottom 0 size "<< bottom.size()<<"  "<<bottom[0]->count()<<" "<<bottom[0]->num()<<std::endl;
  LOG(INFO) << "Bottom 0 size "<< bottom.size()<<"  "<<bottom[0]->shape(0)<<" "<<bottom[0]->shape(1)<<std::endl;

  CHECK_EQ(bottom[0]->num()%T_, 0);
  CHECK_EQ(bottom[0]->count()%bottom[0]->num(), 0);
  CHECK_EQ(bottom[0]->count()/bottom[0]->num(), H_);  // check image vector dimension
  CHECK_EQ(bottom[0]->count(), N_*T_*H_);   // img + sentence length

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } 
  else 
  {
	 
	// this->blobs[0~2] is WLSTM 
	// 3 is We, 4 is be -- for image encoding
	// 5 is Ws, for word encoding
    this->blobs_.resize(3);
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.lstm_param().weight_filler()));
	  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.lstm_param().bias_filler()));
	
	// these 3 parts are WLSTM in Karpathy's code
    // input-to-hidden weights
    // Intialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(4*H_);
    weight_shape.push_back(H_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    weight_shape.clear();
    weight_shape.push_back(4*H_);
    weight_shape.push_back(H_);
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // If necessary, intiialize and fill the bias term
    vector<int> bias_shape(1, 4*H_);
    this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
    bias_filler->Fill(this->blobs_[2].get());
    

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  
  for(int i=0; i<this->blobs_.size(); i++)
  {
    LOG(INFO) << "LSTM Blobs "<<i<<" shape: "<<this->blobs_[i].get()->shape_string() <<std::endl;
  }
  

  //--------------------------------  c[t], h[t], h_to_h ---------------------
  // c[0], h[0], c[t], h[t], h_to_h
  vector<int> cell_shape;
  cell_shape.push_back(N_);
  cell_shape.push_back(H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);
  h_to_h_.Reshape(cell_shape);
  
  
  //--------------------------------  h_to_gate ----------------------------------
  vector<int> gate_shape;
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  h_to_gate_.Reshape(gate_shape);

  LOG(INFO) << "NET PHASE "<<this->phase_;

}

/*
top size: top[0]: (T*N, H)  320x50
Hout_ size:  (T, N, H)      320x1x50
*/
template <typename Dtype>
void LstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  //LOG(INFO) << "~~~~~ T(max sent lengths) "<<T_<<std::endl;
  
  vector<int> original_top_shape;
  original_top_shape.push_back(T_*N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  //LOG(INFO) << "Top size after reshape "<< top.size()<<"  "<<top[0]->count()<<" "<<top[0]->num()<<" -- "<<top[0]->shape_string()<<std::endl;


  // Gate initialization
  vector<int> gate_shape;
  gate_shape.push_back(T_);
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  IFOG_.Reshape(gate_shape);
  IFOGf_.Reshape(gate_shape);
  
  vector<int> Hout_shape;
  Hout_shape.push_back(T_);
  Hout_shape.push_back(N_);
  Hout_shape.push_back(H_);
  C_.Reshape(Hout_shape);
  Hout_.Reshape(Hout_shape);
  Hout_.ShareData(*top[0]);
  Hout_.ShareDiff(*top[0]);

  
  // Set up the bias multiplier  -------------  For WLSTM:  WLSTM = [Wi, Wh, bias]
  // bias_multiplier is ONE vector os size (N*T)
  // it multiplies with real bias (blobs_[2]), to expand bias to matrix, so that could add bias to every row
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
  


  //LOG(INFO) << "===========  Reshape Complete============"<<std::endl;

}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{


  Dtype* Hout_data = Hout_.mutable_cpu_data();

  const Dtype* Wi = this->blobs_[0]->cpu_data();
  const Dtype* Wh = this->blobs_[1]->cpu_data();
  const Dtype* bias = this->blobs_[2]->cpu_data();

  
  Dtype* IFOG_data = IFOG_.mutable_cpu_data();    
  Dtype* IFOGf_data = IFOGf_.mutable_cpu_data();
  Dtype* C_data = C_.mutable_cpu_data();
  Dtype* h_to_gate = h_to_gate_.mutable_cpu_data(); 
  const Dtype* bottom_data = bottom[0]->cpu_data();

  // Initialize previous state
  caffe_set(c_0_.count(), Dtype(0.), c_0_.mutable_cpu_data());
  caffe_set(h_0_.count(), Dtype(0.), h_0_.mutable_cpu_data());
  caffe_set(C_.count(), Dtype(0.), C_data);
  caffe_set(IFOG_.count(), Dtype(0.), IFOG_data);
  caffe_set(IFOGf_.count(), Dtype(0.), IFOGf_data);


  
  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, H_, Dtype(1.),
      bottom_data, Wi, Dtype(0.), IFOG_data);
      

  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, Dtype(1.),
      bias_multiplier_.cpu_data(), bias, Dtype(1.), IFOG_data);
   


  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) 
  {
    Dtype* Hout_t = Hout_data + Hout_.offset(t);
    Dtype* C_t = C_data + C_.offset(t);
    Dtype* IFOG_t = IFOG_data + IFOG_.offset(t);
    Dtype* IFOGf_t = IFOGf_data + IFOGf_.offset(t);
    Dtype* h_to_gate_t = h_to_gate;

    const Dtype* Hout_t_1 = t > 0 ? (Hout_t - Hout_.offset(1)) : h_0_.cpu_data();
    const Dtype* C_t_1 = t > 0 ? (C_t - C_.offset(1)) : c_0_.cpu_data();

    // Hidden-to-hidden propagation
    // Equivalent to Hout[t-1] * WLSTM
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, Dtype(1.), 
        Hout_t_1, Wh, Dtype(0.), h_to_gate);
        
        
    for (int n = 0; n < N_; ++n) 
    {

      const bool cont = (t > 0);

      if (cont) {
        caffe_add(4*H_, IFOG_t, h_to_gate, IFOG_t);
        //with above line, we now get Karpathy's  IFOG[t] = Hin[t] * WLSTM 

      }
      for (int d = 0; d < H_; ++d) 
      {
        // Apply nonlinearity
        IFOGf_t[d] = sigmoid(IFOG_t[d]);
        //IFOGf_t[H_ + d] = cont ? sigmoid(IFOG_t[H_ + d]) : Dtype(0.);
        IFOGf_t[H_ + d] = sigmoid(IFOG_t[H_ + d]);
        IFOGf_t[2*H_ + d] = sigmoid(IFOG_t[2*H_ + d]);
        IFOGf_t[3*H_ + d] = tanh(IFOG_t[3*H_ + d]);

         //LOG(INFO) << IFOG_t[d]<<" --- "<<IFOGf_t[H_ + d] <<" (())) "<<IFOGf_t[3*H_ + d];

        // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
        C_t[d] = IFOGf_t[H_ + d] * C_t_1[d] + IFOGf_t[d] * IFOGf_t[3*H_ + d];
        Hout_t[d] = IFOGf_t[2*H_ + d] * tanh(C_t[d]);
      }
      

      Hout_t += H_;
      C_t += H_;
      C_t_1 += H_;
      IFOG_t += 4*H_;
      IFOGf_t += 4*H_;
      h_to_gate_t += 4*H_;
    }
  }


}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  //LOG(INFO) << "===========  Backward ============"<<std::endl;
    
  const Dtype* top_data = Hout_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const Dtype* Wi = this->blobs_[0]->cpu_data();
  const Dtype* Wh = this->blobs_[1]->cpu_data();
  
  const Dtype* IFOGf_data = IFOGf_.cpu_data();
  const Dtype* C_data = C_.cpu_data();

  Dtype* dHout = Hout_.mutable_cpu_diff();
  Dtype* dIFOG = IFOG_.mutable_cpu_diff();
  Dtype* dIFOGf = IFOGf_.mutable_cpu_diff();
  Dtype* dC = C_.mutable_cpu_diff();
  
  Dtype* dWi = this->blobs_[0]->mutable_cpu_diff();
  Dtype* dWh = this->blobs_[1]->mutable_cpu_diff();
  Dtype* dbh = this->blobs_[2]->mutable_cpu_diff();


  caffe_set(IFOG_.count(), Dtype(0.), dIFOG);
  caffe_set(IFOGf_.count(), Dtype(0.), dIFOGf);
  caffe_set(C_.count(), Dtype(0.), dC);
    

  for (int t = T_-1; t >= 0; --t) 
  {
    Dtype* dHout_t = dHout + Hout_.offset(t);
    Dtype* dC_t = dC + C_.offset(t);
    Dtype* dIFOGf_t = dIFOGf + IFOGf_.offset(t);
    Dtype* dIFOG_t = dIFOG + IFOG_.offset(t);
    Dtype* dHout_t_1 = t > 0 ? dHout + Hout_.offset(t-1) : h_0_.mutable_cpu_diff();
    Dtype* dC_t_1 = t > 0 ? dC + C_.offset(t-1) : c_0_.mutable_cpu_diff();

    const Dtype* C_t = C_data + C_.offset(t);
    const Dtype* C_t_1 = t > 0 ? C_data + C_.offset(t-1) : c_0_.cpu_data();
    const Dtype* IFOGf_t = IFOGf_data + IFOGf_.offset(t);

    for (int n = 0; n < N_; ++n) 
    {

      const bool cont = (t > 0);
      for (int d = 0; d < H_; ++d) 
      {
        const Dtype tanh_c = tanh(C_t[d]);
        dIFOGf_t[2*H_ + d] = dHout_t[d] * tanh_c;
        dC_t[d] += dHout_t[d] * IFOGf_t[2*H_ + d] * (Dtype(1.) - tanh_c * tanh_c);
        dC_t_1[d] = cont ? dC_t[d] * IFOGf_t[H_ + d] : Dtype(0.);
        dIFOGf_t[H_ + d] = cont ? dC_t[d] * C_t_1[d] : Dtype(0.);
        dIFOGf_t[d] = dC_t[d] * IFOGf_t[3*H_ + d];
        dIFOGf_t[3*H_ +d] = dC_t[d] * IFOGf_t[d];
        
        // See cuda file ActivationBackward
        dIFOG_t[d] = dIFOGf_t[d] * IFOGf_t[d] * (Dtype(1.) - IFOGf_t[d]);
        dIFOG_t[H_ + d] = dIFOGf_t[H_ + d] * IFOGf_t[H_ + d]  * (Dtype(1.) - IFOGf_t[H_ + d]);
        dIFOG_t[2*H_ + d] = dIFOGf_t[2*H_ + d] * IFOGf_t[2*H_ + d] * (Dtype(1.) - IFOGf_t[2*H_ + d]);
        dIFOG_t[3*H_ + d] = dIFOGf_t[3*H_ + d] * (Dtype(1.) - IFOGf_t[3*H_ + d] * IFOGf_t[3*H_ + d]);
      }

      // Clip deriviates before nonlinearity
      if (clipping_threshold_ > Dtype(0.)) {
        caffe_bound(4*H_, dIFOG_t, -clipping_threshold_, clipping_threshold_, dIFOG_t);
      }
      
      

      dHout_t += H_;
      C_t += H_;
      C_t_1 += H_;
      dC_t += H_;
      dC_t_1 += H_;
      IFOGf_t += 4*H_;
      dIFOGf_t += 4*H_;
      dIFOG_t += 4*H_;
    }
    
    // Backprop output errors to the previous time step
    // h_to_h_ is a NxH matrices, added to H_ below
    // In Karpathy's code, h_to_h_ is dHin[t,1+d:]
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, 4*H_,
        Dtype(1.), dIFOG_t + IFOG_.offset(t), 
        Wh, Dtype(0.), h_to_h_.mutable_cpu_data());
    
    // dHout[t-1] += dHin[t,1+d:] = h_to_h
    for (int n = 0; n < N_; ++n) {
      const bool cont = (t > 0);
      const Dtype* h_to_h = h_to_h_.cpu_data() + h_to_h_.offset(n);
      if (cont) {
        caffe_add(H_, dHout_t_1, h_to_h, dHout_t_1);
      }
    }
  }

  

 	// this->blobs[0~2] is WLSTM 
	// 3 is We, 4 is be -- for image encoding
	// 5 is Ws, for word encoding
	
  // In Karpathy's code, dWLSTM += np.outer(Hin[t], dIFOG[t])
  // Written in one sentence: dWLSTM = Hin * dIFOG
  // here, [Wi, Wh, bias] = WLSTM, Hin = [caption_input_param, Hout[1~T-1], 1]
  
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, T_*N_, Dtype(1.),
        dIFOG, bottom_data, Dtype(1.), dWi);
  }

  
  if (this->param_propagate_down_[1]) 
  {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, (T_-1)*N_, Dtype(1.),
        dIFOG + IFOG_.offset(1), top_data, Dtype(1.), dWh);
        
  }

  
  if (this->param_propagate_down_[2]) 
  { 
    // Gradient w.r.t. bias
    
    /*
    //equivalent to below gemv
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, Dtype(1.), T_*N_, Dtype(1.),
       dIFOG, bias_multiplier_.cpu_data(), Dtype(0.), dbh);
    */

    caffe_cpu_gemv(CblasTrans, T_*N_,  4*H_, Dtype(1.), dIFOG,
        bias_multiplier_.cpu_data(), Dtype(1.), dbh);

  }
  

  
  // In Karpathy's code, dX[t] = dHin[t,1:1+d]
  // here, dX = dHin[caption_input_param part] = dIFOG * Wi
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, H_, 4*H_, Dtype(1.), dIFOG, Wi, Dtype(0.), bottom[0]->mutable_cpu_diff());
  //LOG(INFO) << "===========  Backward Complete============"<<std::endl;

}

#ifdef CPU_ONLY
STUB_GPU(LstmLayer);
#endif

INSTANTIATE_CLASS(LstmLayer);
REGISTER_LAYER_CLASS(Lstm);

}  // namespace caffe
