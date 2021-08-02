# A GPU Optimizer for Blue-Noise Screen-Space Sampler

## What is this ? 

Hello there ! This application implements the optimization process described in [Lessons Learned and Improvements when Building Screen-Space Samplers with Blue-Noise Error Distribution ](https://unity-grenoble.github.io/website/publication/2021/06/24/sampling_bluenoise_sig21.html), Belcour and Heitz (2021) on the GPU. It permits to optimize tiles of 2D quasi-random sequences (either Owen scrambled Sobol with XOR scrambling or a Rank-1 lattice with Cranley-Patterson Rotations).
A sampling function is provided with the optimized mask. 
Note that this function returns floats in [0, 1] (1 included because of rounding approximations). This might be an important detail if you're using this sample function in a PBRT sampler. 
A Mitsuba v0.6 sampler is also provided at the root of the project (*ldbnsampler.cpp*), so that you only need to copy and paste it along with a mask.h header (resulting from the optimization) in Mitsuba samplers' folder (*src/samplers*). 

## Requirements

The application has three requirements:
 - OpenGL 4.3 for compute shaders
 - OpenMP to speed up all the pre-computations
 - CMake 3.2 for makefiles generation

It also uses [GLFW](https://github.com/glfw/glfw) to generate an OpenGL context and [GLAD](https://github.com/Dav1dde/glad) to load the OpenGL functions. 
GLFW is included in the repository as a submodule, so make sure you get it using ```git submodule update --init``` after cloning the repository if you did not clone using the ```--recursive``` option in the first place.


## Build instructions

### On Windows (with a Visual Studio):

On any version on VS that supports CMake, just open the project directory with the IDE it's really that simple (tested on VS 2019).

### On Linux:

After cloning the repository and its dependencies (see the *Requirements* section), simply do the following in terminal while at the root of the project:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
```

And you should be good to go!


## Usage

On launch, the optimization process starts automatically and you get a preview on the state of the optimization in the GLFW window (each sequence of the mask is used to integrate the same gaussian, the normalized integration results are displayed). 
As the permutations quickly get hard to visualize, the total number of permutations that were applied is constantly updated in the terminal.

The application expects either none or a two parameters. The first parameter is the sample count you want to optimize the mask for. As the sequence used for the optimization is an owen scrambled sobol sequence, it is strongly recommended to set that parameter with a value that is a power of two such that ```0 < n <= 4096```. The default value for the sample count is 16. 
The second parameter is a threshold that must be strictly greather than zero, it is used to stop the optimization and its default value is 15. 

Therefore you can use:
```
./Optimizer 16 15 RANK1 file.samples
```
to launch the optimization for a Rank-1 lattice sequence at 16 samples per sequence and the default halt condition. The resulting samples will be save in an ASCII file `file.samples`.

The optimization is done by pairs of dimensions. The condition that must be fulfilled to halt the optimization for a given pair of dimension is for the number of accepted permutations in a batch of 100 dispatches to be lower than the threshold (each compute shader dispatch attemps 4096 permutations). Note that the process can take several minutes (or even hours!) to complete depending on your GPU.
The application will close when the 12 dimensions are optimized and the scrambling mask (and a sampling function) is exported at the root of the project in a header file (mask.h).


## Sample result
Here is the kind of result that can be obtained at 16 samples per pixel on the *Boxed* scene (rendered with [Mitsuba](http://www.mitsuba-renderer.org)):

| Traditional whitenoise sampler                                                           | [Bluenoise sampler](https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html) |
| ---------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------------------------:|
| <img src="https://i.imgur.com/GkNUQcz.png" alt="Boxed 16spp whitenoise"> | <img src="https://i.imgur.com/mOj1XTK.png" alt="Boxed 16spp bluenoise">                      |

