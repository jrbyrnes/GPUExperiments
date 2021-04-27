#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <string>
#include <chrono>
#include <math.h>
#include <cstring>
#include <array>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

/////////////////////////////////////////////////////////////////
// Helper function to print vector elements
/////////////////////////////////////////////////////////////////
void printVector(const std::string arrayName,
                 const cl_float * arrayData,
                 const unsigned int length)
{
  int numElementsToPrint = (256 < length) ? 256 : length;
  cout << endl << arrayName << ":" << endl;
  for(int i = 0; i < numElementsToPrint; ++i)
    cout << arrayData[i] << " ";
  cout << endl;
}

/////////////////////////////////////////////////////////////////
// Globals
/////////////////////////////////////////////////////////////////
unsigned long long length = (unsigned long long)2147483648*(unsigned long long)4;
int MAXLENGTH = 8192*128;
cl_float * pX   = NULL;
cl_float * pY   = NULL;
cl_float a      = 2.f;

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;

cl::Kernel kernel;
cl::Buffer bufX;
cl::Buffer bufY;

/////////////////////////////////////////////////////////////////
// The saxpy kernel
/////////////////////////////////////////////////////////////////
string kernelStr      =
    "__kernel void saxpy(const global float * x,\n"
    "                    __global float * y,\n"
    "                    const float a,\n"
    "                    int nIterations)\n"
    "{\n"
    "   uint gid = get_global_id(0);\n"
    "   for (int i = 0; i < nIterations; i++) {\n"
    "     y[gid] = a* x[gid] + y[gid];\n"
    "   }\n"
    "}\n";

/////////////////////////////////////////////////////////////////
// Allocate and initialize memory on the host
/////////////////////////////////////////////////////////////////
void initHost(int arraySize)
{
  size_t sizeInBytes = arraySize * sizeof(cl_float);
  pX = (cl_float *) malloc(sizeInBytes);
  if(pX == NULL)
    throw(string("Error: Failed to allocate input memory on host\n"));

  pY = (cl_float *) malloc(sizeInBytes);
  if(pY == NULL)
    throw(string("Error: Failed to allocate input memory on host\n"));

  for(int i = 0; i < arraySize; i++)
  {
    pX[i] = cl_float(i);
    pY[i] = cl_float(length-1-i);
  }

//  printVector("X", pX, arraySize);
//  printVector("Y", pY, arraySize);
}

/////////////////////////////////////////////////////////////////
// Release host memory
/////////////////////////////////////////////////////////////////
void cleanupHost()
{
  if(pX)
  {
    free(pX);
    pX = NULL;
  }
  if(pY != NULL)
  {
    free(pY);
    pY = NULL;
  }
}

int main(int argc, char * argv[])
{
  int workUnitSize, nIterations, arraySize;
  if (length > MAXLENGTH) {
    nIterations = length/MAXLENGTH;
    arraySize = MAXLENGTH;
  }
  else {
    nIterations = 1;
    arraySize = length;
  }

  try
  {
    /////////////////////////////////////////////////////////////////
    // Allocate and initialize memory on the host
    /////////////////////////////////////////////////////////////////
    initHost(arraySize);

    /////////////////////////////////////////////////////////////////
    // Find the platform
    /////////////////////////////////////////////////////////////////
    cl::Platform::get(&platforms);
    std::vector<cl::Platform>::iterator iter;
    for(iter = platforms.begin(); iter != platforms.end(); ++iter)
    {
      if( !strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.") )
      {
        break;
      }
    }

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL context
    /////////////////////////////////////////////////////////////////
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM,
                                     (cl_context_properties)(*iter)(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    /////////////////////////////////////////////////////////////////
    // Detect OpenCL devices
    /////////////////////////////////////////////////////////////////
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    /////////////////////////////////////////////////////////////////
    // Create an OpenCL command queue
    /////////////////////////////////////////////////////////////////
    queue = cl::CommandQueue(context, devices[0]);





    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    /////////////////////////////////////////////////////////////////
    // Create OpenCL memory buffers
    /////////////////////////////////////////////////////////////////
    bufX = cl::Buffer(context,
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_float) * arraySize,
                      pX);
    bufY = cl::Buffer(context,
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_float) * arraySize,
                      pY);

    /////////////////////////////////////////////////////////////////
    // Load CL file, build CL program object, create CL kernel object
    /////////////////////////////////////////////////////////////////
    cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                 kernelStr.length()));
    program = cl::Program(context, sources);
    program.build(devices);
    kernel = cl::Kernel(program, "saxpy");

    /////////////////////////////////////////////////////////////////
    // Set the arguments that will be used for kernel execution
    /////////////////////////////////////////////////////////////////
    kernel.setArg(0, bufX);
    kernel.setArg(1, bufY);
    kernel.setArg(2, a);
    kernel.setArg(3, nIterations);


    /////////////////////////////////////////////////////////////////
    // Enqueue the kernel to the queue
    // with appropriate global and local work sizes
    /////////////////////////////////////////////////////////////////
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(),
    cl::NDRange(arraySize), cl::NDRange(256));

    //size_t result;
    //kernel.getWorkGroupInfo<decltype(result)>((cl::Device)devices[0], CL_KERNEL_WORK_GROUP_SIZE, &result);
    //printf("max workgroup size is %lu", result);


    /////////////////////////////////////////////////////////////////
    // Enqueue blocking call to read back buffer Y
    /////////////////////////////////////////////////////////////////
    queue.enqueueReadBuffer(bufY, CL_TRUE, 0, arraySize *
    sizeof(cl_float), pY);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    printf("Performed a total of %lld operations (2 ^ %d)\n",length, (int)log2(length));

    printVector("Y", pY, length);

    /////////////////////////////////////////////////////////////////
    // Release host resources
    /////////////////////////////////////////////////////////////////
    cleanupHost();
  }
  catch (cl::Error err)
  {
    /////////////////////////////////////////////////////////////////
    // Catch OpenCL errors and print log if it is a build error
    /////////////////////////////////////////////////////////////////
    cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
    if (err.err() == CL_BUILD_PROGRAM_FAILURE)
    {
      string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      cout << "Program Info: " << str << endl;
    }
    cleanupHost();
  }
  catch(string msg)
  {
    cerr << "Exception caught in main(): " << msg << endl;
    cleanupHost();
  }
}
