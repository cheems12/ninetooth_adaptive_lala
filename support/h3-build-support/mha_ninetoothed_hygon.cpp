#include "infinicore/ops/mha.hpp"
#include "infinicore/context/context.hpp"
#include "mha_f32math.h"
#include "mha72.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace infinicore::op::mha_impl::ninetoothed_pilot {
struct Meta { graph::GraphTensor out,q,k,v; float scale; bool causal; };
void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi, float scale, bool causal) {
    std::fputs("NINETOOOTHED_MHA_PLAN",stderr);
    if (alibi) throw std::runtime_error("NineToothed MHA: alibi not supported");
    const auto &qs=q->shape(), &ks=k->shape(), &vs=v->shape(), &os=out->shape();
    for (const auto &x : {q,k,v,out}) { std::fprintf(stderr," MHA_TENSOR dtype=%d shape=",(int)x->dtype()); for (auto d:x->shape()) std::fprintf(stderr,"%lu,",(unsigned long)d); std::fputs(";",stderr); } std::fprintf(stderr," causal=%d scale=%f",(int)causal,scale);
    if(qs.size()!=4 || ks.size()!=4 || vs.size()!=4 || os!=qs || ks!=vs ||
       qs[0]!=ks[0] || qs[3]!=ks[3] || ks[2]==0 || qs[2]%ks[2]!=0 ||
       (qs[3]!=64 && qs[3]!=128 && !(qs[3]==72 && !causal && static_cast<int>(q->dtype())==19)) || (causal && qs[1]!=ks[1]) ||
       q->dtype()!=k->dtype() || q->dtype()!=v->dtype() || q->dtype()!=out->dtype() ||
       (static_cast<int>(q->dtype())!=12 && static_cast<int>(q->dtype())!=19))
        throw std::runtime_error("NineToothed MHA: unsupported shape or dtype");
    return new Meta{graph::GraphTensor(out),graph::GraphTensor(q),graph::GraphTensor(k),graph::GraphTensor(v),scale,causal};
}
struct View {
    uint64_t shape[4]; int64_t strides[4]; NineToothedTensor tensor;
    explicit View(const Tensor &t) {
        const int order[4]={0,2,1,3};
        for(int i=0;i<4;++i){shape[i]=t->shape()[order[i]]; strides[i]=t->strides()[order[i]];}
        tensor={const_cast<std::byte*>(t->data()),shape,strides};
    }
};
void run(void *ptr) {
    auto *p=static_cast<Meta*>(ptr);
    View q(p->q),k(p->k),v(p->v),out(p->out);
    int32_t causal=p->causal, zero=0, one=1;
    double scale=p->scale;
    NineToothedTensor c{&causal,nullptr,nullptr}, z{&zero,nullptr,nullptr}, o{&one,nullptr,nullptr}, s{&scale,nullptr,nullptr};
    auto launch = p->q->shape()[3]==72 ? &launch_mha72 : &launch_mha_f32math;
    int result=launch((void*)context::getStream(),q.tensor,k.tensor,v.tensor,q.tensor,c,s,out.tensor,z,o,
                                  0,(int)p->q->shape()[3],causal,0,1,(int)p->q->dtype(),32,32);
    if(result) throw std::runtime_error("NineToothed MHA launch error: "+std::to_string(result));
}
void cleanup(void **ptr) { delete static_cast<Meta*>(*ptr); *ptr=nullptr; }
static bool registered=[](){
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON,&plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON,&run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON,&cleanup);
    return true;
}();
}
