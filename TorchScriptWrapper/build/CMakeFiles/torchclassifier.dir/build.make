# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build

# Include any dependencies generated for this target.
include CMakeFiles/torchclassifier.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/torchclassifier.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torchclassifier.dir/flags.make

CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o: CMakeFiles/torchclassifier.dir/flags.make
CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o: ../src/torch-classifier-lib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o"
	/opt/elk/1.0/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-elk-linux/aarch64-elk-linux-g++   -mcpu=cortex-a72+crc+crypto   -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security --sysroot=/opt/elk/1.0/sysroots/aarch64-elk-linux  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o -c /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/src/torch-classifier-lib.cpp

CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.i"
	/opt/elk/1.0/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-elk-linux/aarch64-elk-linux-g++   -mcpu=cortex-a72+crc+crypto   -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security --sysroot=/opt/elk/1.0/sysroots/aarch64-elk-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/src/torch-classifier-lib.cpp > CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.i

CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.s"
	/opt/elk/1.0/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-elk-linux/aarch64-elk-linux-g++   -mcpu=cortex-a72+crc+crypto   -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security --sysroot=/opt/elk/1.0/sysroots/aarch64-elk-linux $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/src/torch-classifier-lib.cpp -o CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.s

# Object files for target torchclassifier
torchclassifier_OBJECTS = \
"CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o"

# External object files for target torchclassifier
torchclassifier_EXTERNAL_OBJECTS =

libtorchclassifier.a: CMakeFiles/torchclassifier.dir/src/torch-classifier-lib.cpp.o
libtorchclassifier.a: CMakeFiles/torchclassifier.dir/build.make
libtorchclassifier.a: CMakeFiles/torchclassifier.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtorchclassifier.a"
	$(CMAKE_COMMAND) -P CMakeFiles/torchclassifier.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torchclassifier.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torchclassifier.dir/build: libtorchclassifier.a

.PHONY : CMakeFiles/torchclassifier.dir/build

CMakeFiles/torchclassifier.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torchclassifier.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torchclassifier.dir/clean

CMakeFiles/torchclassifier.dir/depend:
	cd /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build /home/cimil-01/Desktop/StudentProjects/torchscript-interpreter-wrapper/build/CMakeFiles/torchclassifier.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torchclassifier.dir/depend

