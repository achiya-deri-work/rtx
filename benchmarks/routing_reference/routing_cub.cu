// Optional historical CUDA/CUB reference. Not part of production RTX kernels.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

// CUDA conditional bodies forbid allocation, event and host callback nodes,
// including inside nested child graphs. Validate before driver instantiation.
extern "C" int rtx_routing_validate_graph(void* pointer) {
    auto graph = reinterpret_cast<cudaGraph_t>(pointer);
    size_t count = 0;
    cudaError_t error = cudaGraphGetNodes(graph, nullptr, &count);
    if (error != cudaSuccess) return static_cast<int>(error);
    std::vector<cudaGraphNode_t> nodes(count);
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
    if (error != cudaSuccess) return static_cast<int>(error);
    for (auto node : nodes) {
        cudaGraphNodeType type;
        error = cudaGraphNodeGetType(node, &type);
        if (error != cudaSuccess) return static_cast<int>(error);
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            error = cudaGraphChildGraphNodeGetGraph(node, &child);
            if (error != cudaSuccess) return static_cast<int>(error);
            int result = rtx_routing_validate_graph(child);
            if (result) return result;
        } else if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy
                   && type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeEmpty) {
            return static_cast<int>(cudaErrorNotSupported);
        }
    }
    return 0;
}

struct RepairGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraph_t initializer = nullptr;
    cudaGraph_t body = nullptr;
    const int64_t* loads = nullptr;
    const int64_t* phase = nullptr;
    int experts = 0, capacity = 0, rounds = 0;
};

// Populate a conditional body directly instead of nesting the captured graph
// as a child. Keep the complete dependency edges and kernel launch attributes.
// Unsupported node types fail explicitly; never silently discard graph work.
static cudaError_t append_flat_body(cudaGraph_t destination, cudaGraph_t source,
                                   cudaGraphNode_t* last) {
    // Driver-captured kernels contain CUfunction handles, not host entry
    // pointers. Round-trip their parameters through the driver API too.
    decltype(&cuGraphNodeGetParams) get_params = nullptr;
    decltype(&cuGraphAddNode) add_node = nullptr;
    cudaError_t error = cudaGetDriverEntryPoint("cuGraphNodeGetParams",
        reinterpret_cast<void**>(&get_params), cudaEnableDefault);
    if (error != cudaSuccess || !get_params) return cudaErrorNotSupported;
    error = cudaGetDriverEntryPoint("cuGraphAddNode",
        reinterpret_cast<void**>(&add_node), cudaEnableDefault);
    if (error != cudaSuccess || !add_node) return cudaErrorNotSupported;
    size_t count = 0;
    error = cudaGraphGetNodes(source, nullptr, &count);
    if (error != cudaSuccess) return error;
    std::vector<cudaGraphNode_t> nodes(count), copied;
    error = cudaGraphGetNodes(source, nodes.data(), &count);
    if (error != cudaSuccess) return error;
    std::unordered_map<cudaGraphNode_t, cudaGraphNode_t> mapping;
    while (mapping.size() < count) {
        size_t before = mapping.size();
        for (auto node : nodes) {
            if (mapping.count(node)) continue;
            size_t dep_count = 0;
            error = cudaGraphNodeGetDependencies(node, nullptr, nullptr, &dep_count);
            if (error != cudaSuccess) return error;
            std::vector<cudaGraphNode_t> dependencies(dep_count);
            std::vector<cudaGraphEdgeData> edges(dep_count);
            error = cudaGraphNodeGetDependencies(node, dependencies.data(), edges.data(), &dep_count);
            if (error != cudaSuccess) return error;
            bool ready = true;
            for (auto& dependency : dependencies) {
                auto found = mapping.find(dependency);
                if (found == mapping.end()) { ready = false; break; }
                dependency = found->second;
            }
            if (!ready) continue;
            CUgraphNodeParams params{};
            CUresult driver_error = get_params(node, &params);
            if (driver_error != CUDA_SUCCESS) return static_cast<cudaError_t>(driver_error);
            if (params.type != CU_GRAPH_NODE_TYPE_KERNEL && params.type != CU_GRAPH_NODE_TYPE_MEMCPY
                    && params.type != CU_GRAPH_NODE_TYPE_MEMSET && params.type != CU_GRAPH_NODE_TYPE_EMPTY)
                return cudaErrorNotSupported;
            cudaGraphNode_t clone{};
            driver_error = add_node(&clone, destination, dependencies.data(),
                reinterpret_cast<const CUgraphEdgeData*>(edges.data()), dep_count, &params);
            if (driver_error != CUDA_SUCCESS) return static_cast<cudaError_t>(driver_error);
            if (params.type == CU_GRAPH_NODE_TYPE_KERNEL) {
                error = cudaGraphKernelNodeCopyAttributes(clone, node);
                if (error != cudaSuccess) return error;
            }
            mapping[node] = clone;
            copied.push_back(clone);
        }
        if (mapping.size() == before) return cudaErrorInvalidValue;
    }
    return cudaGraphAddEmptyNode(last, destination, copied.data(), copied.size());
}

__global__ void repair_condition(cudaGraphConditionalHandle handle,
                                 const int64_t* loads, const int64_t* phase,
                                 int experts, int capacity, int max_rounds) {
    bool unfinished = false;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        unfinished |= loads[e] != capacity;
    int any_unfinished = __syncthreads_or(unfinished);
    if (threadIdx.x == 0)
        cudaGraphSetConditional(handle, any_unfinished && *phase < max_rounds);
}

static cudaError_t add_check(cudaGraphNode_t* node, cudaGraph_t graph,
                            const cudaGraphNode_t* dependencies, size_t count,
                            cudaGraphConditionalHandle handle, const int64_t* loads,
                            const int64_t* phase, int experts, int capacity, int rounds) {
    void* args[] = {&handle, &loads, &phase, &experts, &capacity, &rounds};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(repair_condition);
    params.gridDim = dim3(1);
    params.blockDim = dim3(256);
    params.kernelParams = args;
    return cudaGraphAddKernelNode(node, graph, dependencies, count, &params);
}

static cudaError_t append_loop(RepairGraph* state, cudaGraph_t graph,
                               const cudaGraphNode_t* dependencies, size_t count,
                               cudaGraphNode_t* last_node) {
    cudaError_t error = cudaSuccess;
    cudaGraphNode_t init_node{}, first_check{}, loop_node{}, body_node{}, final_check{};
    cudaGraphConditionalHandle handle{};
    cudaGraphNodeParams conditional{};
    cudaGraph_t loop_body{};
#define CHECK(call) if ((error = (call)) != cudaSuccess) goto failure
    CHECK(cudaGraphAddChildGraphNode(&init_node, graph, dependencies, count, state->initializer));
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1,
                                           cudaGraphCondAssignDefault));
    CHECK(add_check(&first_check, graph, &init_node, 1, handle,
                    state->loads, state->phase, state->experts, state->capacity, state->rounds));
    conditional.type = cudaGraphNodeTypeConditional;
    conditional.conditional.handle = handle;
    conditional.conditional.type = cudaGraphCondTypeWhile;
    conditional.conditional.size = 1;
    CHECK(cudaGraphAddNode(&loop_node, graph, &first_check, nullptr, 1, &conditional));
    loop_body = conditional.conditional.phGraph_out[0];
    CHECK(append_flat_body(loop_body, state->body, &body_node));
    CHECK(add_check(&final_check, loop_body, &body_node, 1, handle,
                    state->loads, state->phase, state->experts, state->capacity, state->rounds));
    *last_node = loop_node;
    return cudaSuccess;
failure:
    return error;
#undef CHECK
}

extern "C" int bimoe_repair_graph_create(
    void* initializer, void* body, const int64_t* loads, const int64_t* phase,
    int experts, int capacity, int rounds, void** result) {
    int validation = rtx_routing_validate_graph(initializer);
    if (validation) return validation;
    validation = rtx_routing_validate_graph(body);
    if (validation) return validation;
    auto* state = new RepairGraph;
    state->initializer = reinterpret_cast<cudaGraph_t>(initializer);
    state->body = reinterpret_cast<cudaGraph_t>(body);
    state->loads = loads; state->phase = phase;
    state->experts = experts; state->capacity = capacity; state->rounds = rounds;
    cudaGraphNode_t last{};
    cudaError_t error{};
#define CHECK(call) if ((error = (call)) != cudaSuccess) goto failure
    CHECK(cudaGraphCreate(&state->graph, 0));
    CHECK(append_loop(state, state->graph, nullptr, 0, &last));
    CHECK(cudaGraphInstantiateWithFlags(&state->executable, state->graph, 0));
    *result = state;
    return 0;
failure:
    if (state->executable) cudaGraphExecDestroy(state->executable);
    if (state->graph) cudaGraphDestroy(state->graph);
    delete state;
    return static_cast<int>(error);
#undef CHECK
}

extern "C" int bimoe_repair_graph_launch(void* pointer, void* stream) {
    auto* state = static_cast<RepairGraph*>(pointer);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus status{};
    cudaError_t error = cudaStreamIsCapturing(cuda_stream, &status);
    if (error != cudaSuccess) return static_cast<int>(error);
    if (status == cudaStreamCaptureStatusActive) {
        // A graph containing a conditional cannot be launched into a capture.
        // Add its nodes to the capture graph and advance the stream's frontier.
        cudaGraph_t graph{};
        const cudaGraphNode_t* dependencies = nullptr;
        const cudaGraphEdgeData* edges = nullptr;
        size_t count = 0;
        cudaGraphNode_t last{};
        error = cudaStreamGetCaptureInfo(cuda_stream, &status, nullptr, &graph,
                                         &dependencies, &edges, &count);
        if (error != cudaSuccess) return static_cast<int>(error);
        error = append_loop(state, graph, dependencies, count, &last);
        if (error != cudaSuccess) return static_cast<int>(error);
        return static_cast<int>(cudaStreamUpdateCaptureDependencies(
            cuda_stream, &last, nullptr, 1, cudaStreamSetCaptureDependencies));
    }
    return static_cast<int>(cudaGraphLaunch(state->executable, cuda_stream));
}

extern "C" void bimoe_repair_graph_destroy(void* pointer) {
    if (!pointer) return;
    auto* state = static_cast<RepairGraph*>(pointer);
    cudaGraphExecDestroy(state->executable);
    cudaGraphDestroy(state->graph);
    delete state;
}

// Cold compatibility diagnostic: no Torch graph, model tensors or routing work.
extern "C" int rtx_routing_conditional_probe(int device) {
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return static_cast<int>(error);
    cudaGraph_t initializer{}, body{};
    cudaGraphNode_t empty{};
    int64_t* storage = nullptr;
    void* state = nullptr;
    int result = 0;
#define PROBE(call) if ((error = (call)) != cudaSuccess) { result = static_cast<int>(error); goto done; }
    PROBE(cudaMalloc(reinterpret_cast<void**>(&storage), 2*sizeof(int64_t)));
    PROBE(cudaMemset(storage, 0, 2*sizeof(int64_t)));
    PROBE(cudaGraphCreate(&initializer, 0));
    PROBE(cudaGraphAddEmptyNode(&empty, initializer, nullptr, 0));
    PROBE(cudaGraphCreate(&body, 0));
    PROBE(cudaGraphAddEmptyNode(&empty, body, nullptr, 0));
    result = bimoe_repair_graph_create(initializer, body, storage, storage+1, 1, 0, 1, &state);
done:
    if (state) bimoe_repair_graph_destroy(state);
    if (body) cudaGraphDestroy(body);
    if (initializer) cudaGraphDestroy(initializer);
    if (storage) cudaFree(storage);
    return result;
#undef PROBE
}

extern "C" const char* bimoe_repair_error(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

// Stable per-expert quota selection on packed, active proposal segments.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cmath>

// A quantile needs one value, not the entire top-(capacity+1) matrix. Keep
// each independent vector on-chip and write only that selected FP32 value.
template<int Threads,int Items>
__global__ void routing_quantile(const float* matrix,float* output,
                                 int n,int e,int rank,bool column) {
    using Sort=cub::BlockRadixSort<float,Threads,Items>;
    __shared__ typename Sort::TempStorage scratch;
    float values[Items];
    const int length=column ? n : e;
    const int vector=blockIdx.x;
    #pragma unroll
    for(int j=0;j<Items;++j) {
        int element=threadIdx.x+j*Threads;
        values[j]=element<length ? matrix[column ? element*e+vector : vector*e+element] : -CUDART_INF_F;
    }
    Sort(scratch).SortDescending(values);
    #pragma unroll
    for(int j=0;j<Items;++j)
        if(threadIdx.x*Items+j==rank) output[vector]=values[j];
}

template<int Threads>
static cudaError_t launch_routing_quantile(const float* matrix,float* output,
                                         int n,int e,int rank,bool column,cudaStream_t stream) {
    int length=column ? n : e;
    int vectors=column ? e : n;
#define QUANTILE_CASE(Items) if(length<=Threads*Items) { routing_quantile<Threads,Items><<<vectors,Threads,0,stream>>>(matrix,output,n,e,rank,column); return cudaGetLastError(); }
    QUANTILE_CASE(1) QUANTILE_CASE(2) QUANTILE_CASE(4) QUANTILE_CASE(8)
    QUANTILE_CASE(16) QUANTILE_CASE(32)
    if constexpr (Threads<=128) { QUANTILE_CASE(64) }
#undef QUANTILE_CASE
    return cudaErrorNotSupported;
}

extern "C" int rtx_routing_quantile(const float* matrix,float* output,
                                  int n,int e,int rank,int threads,int column,void* stream) {
    auto s=reinterpret_cast<cudaStream_t>(stream);
    if(threads==64) return int(launch_routing_quantile<64>(matrix,output,n,e,rank,column,s));
    if(threads==128) return int(launch_routing_quantile<128>(matrix,output,n,e,rank,column,s));
    if(threads==256) return int(launch_routing_quantile<256>(matrix,output,n,e,rank,column,s));
    return int(cudaErrorInvalidValue);
}

__global__ void quota_init(int* counts, bool* accepted, int m, int e) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<e) counts[i]=0;
    if (i<m) accepted[i]=false;
}

__global__ void quota_count(const int64_t* groups, const float* gains,
                             const int64_t* quotas, int* counts, int m, int e,
                             const bool* filter) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=m) return;
    int64_t g=groups[i];
    // -inf sorts after every finite priority and is never accepted. NaN and
    // +inf must remain: their ranks consume quota even though isfinite fails.
    if (g>=0 && g<e && quotas[g]>0 && gains[i]!=-CUDART_INF_F && (!filter || filter[i]))
        atomicAdd(counts+g,1);
}

__global__ void quota_prefix(int* counts, int* offsets, int e) {
    using Scan=cub::BlockScan<int,256>;
    __shared__ typename Scan::TempStorage scratch;
    int input[4], output[4], total;
    #pragma unroll
    for (int j=0;j<4;++j) {
        int i=threadIdx.x*4+j;
        input[j]=i<e ? counts[i] : 0;
    }
    Scan(scratch).ExclusiveSum(input,output,total);
    __syncthreads();
    #pragma unroll
    for (int j=0;j<4;++j) {
        int i=threadIdx.x*4+j;
        if (i<e) { offsets[i]=output[j]; counts[i]=0; }
    }
    if (threadIdx.x==0) offsets[e]=total;
}

__device__ uint64_t quota_key(float gain, uint32_t index) {
    // Canonicalize +/-zero and NaN payloads to preserve floating stable-sort
    // equality. Low bits make original proposal order independent of atomics.
    uint32_t bits=__float_as_uint(gain==0.f ? 0.f : gain);
    uint32_t ordered=isnan(gain) ? 0xffffffffu :
        ((bits&0x80000000u) ? ~bits : (bits^0x80000000u));
    return (uint64_t(ordered)<<32) | uint64_t(0xffffffffu-index);
}

__global__ void quota_scatter(const int64_t* groups, const float* gains,
                               const int64_t* quotas, int* cursors,
                               const int* offsets, uint64_t* keys, int m, int e,
                               const bool* filter) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=m) return;
    int64_t g=groups[i];
    if (g>=0 && g<e && quotas[g]>0 && gains[i]!=-CUDART_INF_F && (!filter || filter[i])) {
        int position=atomicAdd(cursors+g,1);
        keys[offsets[g]+position]=quota_key(gains[i],i);
    }
}

__global__ void quota_finish(const uint64_t* keys, const int* offsets,
                              const int64_t* quotas, const float* gains,
                              bool* accepted) {
    int g=blockIdx.x;
    int start=offsets[g], count=offsets[g+1]-start;
    int64_t take=quotas[g];
    if (take>count) take=count;
    for (int rank=threadIdx.x;rank<take;rank+=blockDim.x) {
        uint32_t i=0xffffffffu-uint32_t(keys[start+rank]);
        if (isfinite(gains[i])) accepted[i]=true;
    }
}

extern "C" int bimoe_quota_workspace(int m,int e,size_t* bytes) {
    return int(cub::DeviceSegmentedRadixSort::SortKeysDescending(
        nullptr,*bytes,(const uint64_t*)nullptr,(uint64_t*)nullptr,m,e,
        (const int*)nullptr,(const int*)nullptr,0,64));
}

extern "C" int bimoe_quota_select(
    const int64_t* groups,const float* gains,const int64_t* quotas,
    int* counts,int* offsets,uint64_t* keys,uint64_t* sorted,
    void* workspace,size_t bytes,bool* accepted,int m,int e,void* stream_ptr) {
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    int blocks=((m>e ? m : e)+255)/256;
    quota_init<<<blocks,256,0,stream>>>(counts,accepted,m,e);
    quota_count<<<(m+255)/256,256,0,stream>>>(groups,gains,quotas,counts,m,e,nullptr);
    quota_prefix<<<1,256,0,stream>>>(counts,offsets,e);
    quota_scatter<<<(m+255)/256,256,0,stream>>>(groups,gains,quotas,counts,offsets,keys,m,e,nullptr);
    cudaError_t error=cudaGetLastError();
    if (error!=cudaSuccess) return int(error);
    error=cub::DeviceSegmentedRadixSort::SortKeysDescending(
        workspace,bytes,keys,sorted,m,e,offsets,offsets+1,0,64,stream);
    if (error!=cudaSuccess) return int(error);
    quota_finish<<<e,256,0,stream>>>(sorted,offsets,quotas,gains,accepted);
    return int(cudaGetLastError());
}

__global__ void pair_init(int* counts,bool* source_mask,bool* target_mask,
                           const int64_t* loads,const int64_t* phase,
                           int64_t* new_loads,int64_t* new_phase,int m,int e) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<e) { counts[i]=0;counts[e+i]=0;new_loads[i]=loads[i]; }
    if(i<m) {source_mask[i]=false;target_mask[i]=false;}
    if(i==0) *new_phase=*phase+1;
}

__global__ void pair_apply(const int64_t* sources,const int64_t* targets,
                            const int64_t* slots,const bool* accepted,
                            int64_t* result,int64_t* loads,int m,int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=m) return;
    int token=i/k;
    result[token*k+slots[i]]=accepted[i] ? targets[i] : sources[i];
    if(accepted[i]) {
        atomicAdd(reinterpret_cast<unsigned long long*>(loads+sources[i]),~0ull);
        atomicAdd(reinterpret_cast<unsigned long long*>(loads+targets[i]),1ull);
    }
}

extern "C" int bimoe_quota_apply(
    const int64_t* sources,const int64_t* targets,const float* gains,
    const int64_t* surplus,const int64_t* deficit,const int64_t* slots,
    const int64_t* loads,const int64_t* phase,int* counts,int* offsets,
    uint64_t* keys,uint64_t* sorted,void* workspace,size_t bytes,
    bool* source_mask,bool* target_mask,int64_t* result,int64_t* new_loads,
    int64_t* new_phase,int m,int e,int k,void* stream_ptr) {
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    int blocks=((m>e ? m : e)+255)/256;
    pair_init<<<blocks,256,0,stream>>>(counts,source_mask,target_mask,loads,phase,new_loads,new_phase,m,e);
    for(int stage=0;stage<2;++stage) {
        auto groups=stage==0 ? sources : targets;
        auto quotas=stage==0 ? surplus : deficit;
        auto cursor=counts+stage*e;
        auto filter=stage==0 ? nullptr : source_mask;
        auto accepted=stage==0 ? source_mask : target_mask;
        quota_count<<<(m+255)/256,256,0,stream>>>(groups,gains,quotas,cursor,m,e,filter);
        quota_prefix<<<1,256,0,stream>>>(cursor,offsets,e);
        quota_scatter<<<(m+255)/256,256,0,stream>>>(groups,gains,quotas,cursor,offsets,keys,m,e,filter);
        auto error=cub::DeviceSegmentedRadixSort::SortKeysDescending(
            workspace,bytes,keys,sorted,m,e,offsets,offsets+1,0,64,stream);
        if(error!=cudaSuccess) return int(error);
        quota_finish<<<e,256,0,stream>>>(sorted,offsets,quotas,gains,accepted);
    }
    pair_apply<<<(m+255)/256,256,0,stream>>>(sources,targets,slots,target_mask,result,new_loads,m,k);
    return int(cudaGetLastError());
}

extern "C" const char* bimoe_quota_error(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

extern "C" int rtx_routing_copy(void* dst,const void* src,size_t bytes,void* stream) {
 return int(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,reinterpret_cast<cudaStream_t>(stream)));
}
