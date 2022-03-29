# Deep Learning Runtime wrappers for classification on ElkOS

Classification wrappers that expose the same API for the following Deep Learning Embedded Runtimes:
- TensorFlow Lite
- Torchscript/Pytorch C++
- ONNX Runtime
- RTNeural

Each folder contains scripts and library binaries to compile libraries for the [Elk Audio OS](https://github.com/elk-audio)

[**tensorflow_model_conversion.ipynb**](https://github.com/domenicostefani/deep-classf-runtime-wrappers/blob/master/tensorflow_model_conversion.ipynb) contains utilities to convert a TensorFlow model to the formats accepted by each runtime.

_Domenico Stefani, Simone Peroni 2022_
