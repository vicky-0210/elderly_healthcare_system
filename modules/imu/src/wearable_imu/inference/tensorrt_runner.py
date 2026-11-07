"""Optional fixed-shape TensorRT 8 execution on Jetson Nano (one context per process)."""
import numpy as np


class TensorRTWindow:
    def __init__(self,path,channels,width):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # create a CUDA context for the standalone inference process
        self.cuda=cuda
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None or self.engine.num_bindings!=2: raise ValueError("Invalid two-binding TensorRT 8 engine")
        self.ctx=self.engine.create_execution_context()
        self.i=self.engine.get_binding_index("window"); self.o=self.engine.get_binding_index("logit")
        if self.i<0 or self.o<0 or tuple(self.engine.get_binding_shape(self.i))!=(1,channels,width):
            raise ValueError("Engine input does not match the selected export")
        self.input=np.empty((1,channels,width),dtype=trt.nptype(self.engine.get_binding_dtype(self.i)))
        self.output=np.empty(1,dtype=trt.nptype(self.engine.get_binding_dtype(self.o)))
        self.src=cuda.mem_alloc(self.input.nbytes); self.dst=cuda.mem_alloc(self.output.nbytes)
        self.bindings=[0,0]; self.bindings[self.i]=int(self.src); self.bindings[self.o]=int(self.dst)

    def __call__(self,normalized):
        out=[]
        for window in normalized:
            self.input[0]=window.T
            self.cuda.memcpy_htod(self.src,self.input)
            if not self.ctx.execute_v2(self.bindings): raise RuntimeError("TensorRT execution failed")
            self.cuda.memcpy_dtoh(self.output,self.dst); out.append(float(self.output[0]))
        return np.asarray(out)
