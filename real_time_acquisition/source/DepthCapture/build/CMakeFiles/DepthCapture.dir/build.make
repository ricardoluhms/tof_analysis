# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build

# Include any dependencies generated for this target.
include CMakeFiles/DepthCapture.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DepthCapture.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DepthCapture.dir/flags.make

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o: CMakeFiles/DepthCapture.dir/flags.make
CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o: ../DepthCapture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o -c /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/DepthCapture.cpp

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DepthCapture.dir/DepthCapture.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/DepthCapture.cpp > CMakeFiles/DepthCapture.dir/DepthCapture.cpp.i

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DepthCapture.dir/DepthCapture.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/DepthCapture.cpp -o CMakeFiles/DepthCapture.dir/DepthCapture.cpp.s

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.requires:

.PHONY : CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.requires

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.provides: CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.requires
	$(MAKE) -f CMakeFiles/DepthCapture.dir/build.make CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.provides.build
.PHONY : CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.provides

CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.provides.build: CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o


# Object files for target DepthCapture
DepthCapture_OBJECTS = \
"CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o"

# External object files for target DepthCapture
DepthCapture_EXTERNAL_OBJECTS =

DepthCapture: CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o
DepthCapture: CMakeFiles/DepthCapture.dir/build.make
DepthCapture: /usr/lib/libvoxel.so.0.6.10
DepthCapture: CMakeFiles/DepthCapture.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DepthCapture"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DepthCapture.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DepthCapture.dir/build: DepthCapture

.PHONY : CMakeFiles/DepthCapture.dir/build

CMakeFiles/DepthCapture.dir/requires: CMakeFiles/DepthCapture.dir/DepthCapture.cpp.o.requires

.PHONY : CMakeFiles/DepthCapture.dir/requires

CMakeFiles/DepthCapture.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DepthCapture.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DepthCapture.dir/clean

CMakeFiles/DepthCapture.dir/depend:
	cd /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build /home/vinicius/utils/tof/real_time_acquisition/source/DepthCapture/build/CMakeFiles/DepthCapture.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DepthCapture.dir/depend

