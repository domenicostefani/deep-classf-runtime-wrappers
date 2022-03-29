# Deep Learning Runtime wrappers for Embedded Real-time Audio Classification

Classification wrappers that expose the same API for the following Deep Learning Embedded Runtimes:
- TensorFlow Lite
- Torchscript/Pytorch C++
- ONNX Runtime
- RTNeural

Each folder contains scripts and library binaries to compile libraries for the [Elk Audio OS](https://github.com/elk-audio)

## API functions
```
/** Dynamically allocate an instance of a classifier object (do not use in real time threads!) */
ClassifierPtr createClassifier(const std::string &filename, bool verbose = false);

/** Feed a feature array (C Array) to the model, perform inference and return the prediction */
int classify(ClassifierPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

template<std::size_t IN_SIZE, std::size_t OUT_SIZE>
int classify(ClassifierPtr cls, std::array<float,IN_SIZE>& featureArray, std::array<float,OUT_SIZE>& outputArray);

/** Free the classifier memory (do not use in real time threads) */
void deleteClassifier(ClassifierPtr cls);
```

## Real-time Safety

The `classify(...)` function is meant to be called from real-time threads (e.g., audio thread) so it is meant to be real-time safe, if the interpreter at hand allows for rt-safe inference.
RT-safety was tested for:
- [x] **TensorFlow Lite**
- [x] **TorchScript** (Yes, but classify must be called once during the first call of the audio processing function, otherwise it allocates memory. See [this](https://forum.elk.audio/t/allocation-evades-sigxcpu/))
- [x] **Onnx Runtime** (Yes, but same as TorchScript. See [this](https://forum.elk.audio/t/allocation-evades-sigxcpu/))
- [x] **RtNeural**

The `createClassifier(...)` and `deleteClassifier(...)` functions will mess with memory allocation and are definitely not meant to be called from real-time thread, but only once on program start and end.

## Model Conversion Utils

The interactive notebook [tensorflow_model_conversion.ipynb](https://github.com/domenicostefani/deep-classf-runtime-wrappers/blob/master/tensorflow_model_conversion.ipynb) contains utilities to convert a TensorFlow model to the formats accepted by each runtime.

_Domenico Stefani, Simone Peroni 2022_
