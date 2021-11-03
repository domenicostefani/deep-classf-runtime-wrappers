# tflite-interpreter-wrapper
Wrapper for the TensorFlow Lite interpreter

### Files and folders:

- [test_library/](https://github.com/domenicostefani/tflite-interpreter-wrapper/tree/master/test_library) contains a simple cpp to test that the compiled library is working.
- [the_compiled_library/](https://github.com/domenicostefani/tflite-interpreter-wrapper/tree/master/the_compiled_library) contains the compiled binaries for the library.
- [your_library_here/](https://github.com/domenicostefani/tflite-interpreter-wrapper/tree/master/your_library_here) contains the source code for the library.
- [compile_static_lib_aarch64.sh](https://github.com/domenicostefani/tflite-interpreter-wrapper/blob/master/compile_static_lib_aarch64.sh) is a bash script used to compile the library source code into a static library (`.a` file) FOR AN ARM ARCH64 computer.
- [compile_static_lib_x86_64.sh](https://github.com/domenicostefani/tflite-interpreter-wrapper/blob/master/compile_static_lib_x86_64.sh) is a bash script used to compile the library source code into a static library (`.a` file) FOR A REGULAR x86/64 computer.

---

### How to modify and compile:
1. Modify the inner workings of the library by changing the code in `your_library_here/`,
2. Run `compile_static_lib_x86_64.sh` to compile for a regular Intel/Amd computer (or the other script for the RaspberryPI4),
3. Take `libliteclassifier.a` and the header`libliteclassifier.h` from `the_compiled_library/` folder.

### To test the Library:
1. go to `test_library/`,
2. If needed edit the `main.cpp file`,
3. Compile `main.cpp` along with the library with `test_library/compile_with_lib_x86_64.sh` or `test_library/compile_with_lib_aarch64.sh`
4. Test the compiled library by running the compiled executable and passing the path to the deep learning model (e.g., `./inference_x86_64 test_model.tflite `).
