"""cuda-python host orchestration of CuTe/Inductor GPU routing graphs.

No native extension build. Kernel code, including the conditional predicate,
is compiled by CuTe DSL; this module only constructs/submits CUDA graphs.
"""
import ctypes

from cuda.bindings import driver as cuda


def checked(result):
    if result[0]!=cuda.CUresult.CUDA_SUCCESS:
        detail=cuda.cuGetErrorString(result[0])
        message=detail[1] if detail[0]==cuda.CUresult.CUDA_SUCCESS else ''
        raise RuntimeError(f'Routing CUDA driver error: {result[0]}: {message}')
    if len(result)==1:return None
    return result[1] if len(result)==2 else result[1:]


def nodes(graph):
    _,count=checked(cuda.cuGraphGetNodes(graph))
    return checked(cuda.cuGraphGetNodes(graph,count))[0]


def dependencies(node):
    _,_,count=checked(cuda.cuGraphNodeGetDependencies(node))
    deps,edges,_=checked(cuda.cuGraphNodeGetDependencies(node,count))
    return deps,edges


def append_flat_body(destination,source):
    mapping={};pending=list(nodes(source))
    allowed=(cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL,
             cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY,
             cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET,
             cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY)
    while pending:
        progress=False
        for node in tuple(pending):
            deps,edges=dependencies(node)
            if any(int(d) not in mapping for d in deps):continue
            params=checked(cuda.cuGraphNodeGetParams(node))
            if params.type not in allowed:
                raise RuntimeError(f'Unsupported routing conditional-body node: {params.type}')
            mapped=[mapping[int(d)] for d in deps]
            clone=checked(cuda.cuGraphAddNode(destination,mapped or None,edges or None,len(mapped),params))
            if params.type==cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                checked(cuda.cuGraphKernelNodeCopyAttributes(clone,node))
            mapping[int(node)]=clone;pending.remove(node);progress=True
        if not progress:raise RuntimeError('Unresolved routing graph dependencies')
    copies=list(mapping.values())
    return checked(cuda.cuGraphAddEmptyNode(destination,copies or None,len(copies)))


class ConditionalRoutingGraph:
    def __init__(self,initializer,body,loads,phase,e,capacity,rounds,device):
        import torch
        import cutlass
        from .kernels.routing_selection import compile_condition
        required=('cuGraphNodeGetParams','cuGraphConditionalHandleCreate','cuGraphAddNode',
                  'cuStreamUpdateCaptureDependencies','CUstreamUpdateCaptureDependencies_flags')
        missing=[name for name in required if not hasattr(cuda,name)]
        if missing:raise RuntimeError('BalancedRouter needs CUDA Python conditional-graph APIs '
            f'({missing}); install rtx-blackwell[routing]. Tested cuda-python: 13.3.1.')
        self.device=device;self.graph=None;self.executable=None;self.context=None
        self.initializer=cuda.CUgraph(initializer.raw_cuda_graph())
        self.body=cuda.CUgraph(body.raw_cuda_graph())
        self.context=checked(cuda.cuDevicePrimaryCtxRetain(device))
        self.condition=compile_condition(e,capacity,rounds)
        temporary=checked(cuda.cuGraphCreate(0))
        try:
            handle=checked(cuda.cuGraphConditionalHandleCreate(temporary,self.context,1,1))
            self._add_conditional(temporary,[],handle)
            # Conditional setters execute only as graph nodes. Compilation
            # above loads the kernel; never execute a setter as a warmup launch.
            capture=torch.cuda.Stream(device=device)
            capture.wait_stream(torch.cuda.current_stream(device))
            self.condition_template=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.condition_template,stream=capture):
                self.condition(loads,phase.view(-1),cutlass.Uint64(int(handle)))
            capture.synchronize()
            template_nodes=nodes(cuda.CUgraph(self.condition_template.raw_cuda_graph()))
            if len(template_nodes)!=1:raise RuntimeError('Expected one CuTe conditional-predicate node')
            self.condition_node=template_nodes[0]
        finally:checked(cuda.cuGraphDestroy(temporary))
        self.graph=checked(cuda.cuGraphCreate(0))
        self.append(self.graph,[])
        self.executable=checked(cuda.cuGraphInstantiate(self.graph,0))

    def _add_conditional(self,graph,deps,handle):
        params=cuda.CUgraphNodeParams()
        params.type=cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle=handle
        params.conditional.type=cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        params.conditional.size=1
        params.conditional.ctx=self.context
        node=checked(cuda.cuGraphAddNode(graph,deps or None,None,len(deps),params))
        return node,params.conditional.phGraph_out[0]

    def _add_check(self,graph,deps,handle):
        params=checked(cuda.cuGraphNodeGetParams(self.condition_node))
        # This private CuTe kernel's ABI is (loads*, phase*, uint64 handle).
        # Copy pointer addresses, never mutate the driver's original storage.
        previous=ctypes.cast(params.kernel.kernelParams,ctypes.POINTER(ctypes.c_void_p))
        value=ctypes.c_uint64(int(handle))
        arguments=(ctypes.c_void_p*3)(previous[0],previous[1],ctypes.addressof(value))
        params.kernel.kernelParams=ctypes.addressof(arguments)
        result=checked(cuda.cuGraphAddNode(graph,deps or None,None,len(deps),params))
        checked(cuda.cuGraphKernelNodeCopyAttributes(result,self.condition_node))
        return result

    def append(self,graph,deps):
        initial=checked(cuda.cuGraphAddChildGraphNode(graph,deps or None,len(deps),self.initializer))
        handle=checked(cuda.cuGraphConditionalHandleCreate(graph,self.context,1,1))
        first=self._add_check(graph,[initial],handle)
        loop,body=self._add_conditional(graph,[first],handle)
        last=append_flat_body(body,self.body)
        self._add_check(body,[last],handle)
        return loop

    def launch(self,stream):
        stream=cuda.CUstream(stream)
        status,_,graph,deps,_,_=checked(cuda.cuStreamGetCaptureInfo(stream))
        if status==cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            last=self.append(graph,deps)
            replace=cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES
            checked(cuda.cuStreamUpdateCaptureDependencies(stream,[last],None,1,replace))
        else:checked(cuda.cuGraphLaunch(self.executable,stream))

    def close(self):
        if self.executable is not None:
            checked(cuda.cuGraphExecDestroy(self.executable));self.executable=None
        if self.graph is not None:checked(cuda.cuGraphDestroy(self.graph));self.graph=None
        if self.context is not None:checked(cuda.cuDevicePrimaryCtxRelease(self.device));self.context=None

    def __del__(self):
        try:self.close()
        except Exception:pass


def copy_async(dst,src,bytes,stream):
    checked(cuda.cuMemcpyDtoDAsync(cuda.CUdeviceptr(dst),cuda.CUdeviceptr(src),bytes,cuda.CUstream(stream)))
