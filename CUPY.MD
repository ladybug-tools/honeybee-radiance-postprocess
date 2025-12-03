# CuPy installation guide

## :white_check_mark: CuPy requirements

* You have an NVIDIA GPU.
* Your GPU's compute capability is supported by CuPy.
* You have a suitable CUDA Toolkit installed (matching the CuPy package variant).
* Your GPU driver supports the CUDA version you intend to use.

## :wrench: Compute Capability

### What is Compute Capability
Compute capability is a number that describes the feature set and performance characteristics of your NVIDIA GPU architecture.

CuPy supports GPUs with compute capability ≥ 3.0.

Examples:
* NVIDIA GTX 1080 &rarr; Compute Capability **6.1**.
* NVIDIA GTX 2080 &rarr; Compute Capability **7.5**.
* NVIDIA RTX 3080 &rarr; Compute Capability **8.6**.
* NVIDIA RTX 4090 &rarr; Compute Capability **8.9**.
* NVIDIA RTX 5090 &rarr; Compute Capability **12.0**.

You can find the compute capability of your GPU [here](https://developer.nvidia.com/cuda-gpus). If you have an older GPU, such as the GTX 10 series or older, please check [this](https://developer.nvidia.com/cuda-legacy-gpus) list.

If you are still not sure about your compute capability, you can query it through CuPy once it is installed:

```python
import cupy as cp

print(cp.cuda.Device().compute_capability)
```

This prints the major index and minor index as a string, e.g., '75' for version 7.5.

## :toolbox: CUDA Toolkit

### What is CUDA Toolkit
The CUDA Toolkit provides the underlying CUDA libraries, runtime, and compiler that CuPy uses to run GPU code. CuPy depends on a matching CUDA Toolkit and GPU driver that supports your GPU.

### Picking the correct CUDA version

You can find the CUDA version supported by your GPU driver by executing  [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) in the command prompt.

```
C:\Users\USERNAME>nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.80                 Driver Version: 581.80         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
```

Please note that the CUDA version reported by nvidia-smi is not the installed version of the CUDA Toolkit; it is the latest API compatible version for your GPU driver. If your GPU driver is not up to date you might not see the latest CUDA version.

### Installing CUDA Toolkit

You can download the latest version of CUDA Toolkit [here](https://developer.nvidia.com/cuda-downloads). If your GPU driver does not support the latest version, you can select a compatible version in the [archive](https://developer.nvidia.com/cuda-toolkit-archive).

## :package: Installing CuPy

CuPy provides prebuilt binary wheels tied to CUDA versions. You should install the wheel matching your installed CUDA Toolkit.

| CUDA Toolkit	      | Command                  |
| ------------------- | ------------------------ |
| CUDA 13.x	          | pip install cupy-cuda13x |
| CUDA 12.x      	  | pip install cupy-cuda12x |
| CUDA 11.x (>= 11.2) |	pip install cupy-cuda11x |

## :question: FAQ

### Can I use CuPy with an AMD GPU?

If you have an AMD GPU please read [this](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental).

### How can I tell if honeybee-radiance-process is using CuPy and not NumPy?

honeybee-radiance-postprocess will print a line that tells you if CuPy is being used. You should see something like this:

`Using CuPy (13.6.0) for GPU (NVIDIA GeForce RTX 3080) acceleration in honeybee-radiance-postprocess.`

If CuPy *is* installed but fails during initialization honeybee-radiance-postprocess will fall back to NumPy. If this happens you should see a statement similar to this:

`Failed to load CuPy successfully. Falling back to NumPy (1.26.4) in honeybee-radiance-postprocess`

## :warning: Common Errors

`FileNotFoundError: Could not find module 'nvrtc64_130_0.dll' (or one of its dependencies). Try using the full path with constructor syntax.`

This is possibly caused by a mismatch between the CUDA Toolkit version and the CuPy version, e.g., cupy-cuda13x and CUDA Toolkit 12.9.
