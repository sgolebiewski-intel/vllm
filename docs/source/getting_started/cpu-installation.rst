.. _installation_cpu:

Installation with CPU
========================

vLLM initially supports basic model inference and serving on x86 CPU platform, with FP32 and BF16 data types .

Table of contents:

- :ref:`Requirements <cpu_backend_requirements>`
- :ref:`Quick start using Dockerfile <cpu_backend_quick_start_dockerfile>`
- :ref:`Build from source <build_cpu_backend_from_source>`
- :ref:`Intel Extension for PyTorch <ipex_guidance>`
- :ref:`Performance tips <cpu_backend_performance_tips>`

.. _cpu_backend_requirements:

Requirements
------------

* OS: Linux
* Compiler: gcc/g++>=12.3.0 (optional, recommended)
* Instruction set architecture (ISA) requirement: AVX512 is required.

.. _cpu_backend_quick_start_dockerfile:

Quick start using Dockerfile
----------------------------

.. code-block:: console

   $ docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
   $ docker run -it \
                --rm \
                --network=host \
                --cpuset-cpus=<cpu-id-list, optional> \
                --cpuset-mems=<memory-node, optional> \
                vllm-cpu-env

.. _build_cpu_backend_from_source:

Build from source
-----------------

- First, install recommended compiler. We recommend to use ``gcc/g++ >= 12.3.0`` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

   $ sudo apt-get update  -y
   $ sudo apt-get install -y gcc-12 g++-12
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

- Second, install Python packages for vLLM CPU backend building:

.. code-block:: console

   $ pip install --upgrade pip
   $ pip install wheel packaging ninja "setuptools>=49.4.0" numpy
   $ pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

- Finally, build and install vLLM CPU backend:

.. code-block:: console

   $ VLLM_TARGET_DEVICE=cpu python setup.py install

.. note::

   - BF16 is the default data type in the current CPU backend (that means the backend will
     cast FP16 to BF16), and is compatible with all CPUs with AVX512 ISA support.
   - AVX512_BF16 is an extension that ISA provides, with native BF16 data type conversion
     and vector product instructions. It brings some performance improvement compared to
     pure AVX512. The CPU backend build script will check the host CPU flags to
     determine whether to enable AVX512_BF16 or not.
   - If you want to force enable AVX512_BF16 for the cross-compilation, set the
     ``VLLM_CPU_AVX512BF16=1`` environment variable before building.

.. _ipex_guidance:

Intel Extension for PyTorch
---------------------------

- `Intel Extension for PyTorch (IPEX) <https://github.com/intel/intel-extension-for-pytorch>`_
  extends PyTorch with up-to-date feature optimizations for an extra performance boost on Intel hardware.

- IPEX after the ``2.3.0`` version can be enabled in the CPU backend by default if it is installed.

.. _cpu_backend_performance_tips:

Performance tips
-----------------

- vLLM CPU backend uses the ``VLLM_CPU_KVCACHE_SPACE`` environment variable to
  specify the KV Cache size (e.g, ``VLLM_CPU_KVCACHE_SPACE=40`` means 40 GB space
  for KV cache), higher setting will enable vLLM to run more requests in parallel.
  This parameter should be set based on the hardware configuration and memory
  management pattern of users.

- We highly recommend to use TCMalloc for high performance memory allocation and
  better cache locality. For example, on Ubuntu 22.4, you can run:

.. code-block:: console

   $ sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
   $ find / -name *libtcmalloc* # find the dynamic link library path
   $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
   $ python examples/offline_inference.py # run vLLM

- vLLM CPU backend uses OpenMP for thread-parallel computation. If you want the
  best performance on CPU, it is crucial to isolate CPU cores for OpenMP threads
  with other thread pools (like web-service event-loop), to avoid CPU oversubscription.

- It is recommended to disable the hyper-threading when using vLLM CPU backend on a bare-metal machine.

- If you are using vLLM CPU backend on a multi-socket machine with NUMA, make
  sure to set CPU cores and memory nodes, to avoid the remote memory node access.
  ``numactl`` is a useful tool for CPU core and memory binding on NUMA platform.
  Additionaly, the ``--cpuset-cpus`` and ``--cpuset-mems`` arguments of ``docker run``
  are also useful.



