#include <cuda.h>
#include <chrono>

#include "../lib/tree.hpp"

__global__ void FIND(int *, int *, int *);

int main(int argc, char *argv[])
{
    // std::srand((unsigned)std::time(NULL)); // reset random seed
    int levels = 19; // level depth of tree
    int maxChildren = 2; // maximum number of children a node can have
    int maxNodeValue = 1999; // maximum data value that can be stored in a node
    bool fullyBalanced = false; // whether to generate a random or fully balanced tree

    // check passed arguments
    levels = argc >= 2 ? std::stoi(argv[1]) : levels;
    maxChildren = argc >= 3 ? std::stoi(argv[2]) : maxChildren;
    maxNodeValue = argc >= 4 ? std::stoi(argv[3]) : maxNodeValue;
    fullyBalanced = argc = 5 ? std::stoi(argv[4]) : fullyBalanced;

    int l = 0;
    long int maxNodes = 0; // maximum number of nodes that tree can contain
    // calculate maxNodes
    for(; l <= levels; ++l)
    {
        maxNodes += pow(maxChildren, l);
    }

    int *dataArray; // array for storing tree data
    try
    {
        dataArray = (int *)malloc(maxNodes * sizeof(int)); // allocate memory for tree's data based off of maxNodes as repeated calls for realloc is time consuming
    }
    catch(const std::exception& e)
    {
        std::cerr << "Could not allocate memory for dataArray.\n" << e.what() << "\n";
        exit(EXIT_FAILURE);
    }
    auto root = tree::GenerateTree(dataArray, &levels, &maxChildren, &maxNodeValue, &fullyBalanced); // constuct tree with specified parameters
    std::cout << "Tree generated with " << nodeCount << " nodes.\n";

    int searchValue; // value to search for inside tree
    std::cout << "Enter search value: ";
    std::cin >> searchValue;

    int factor = tree::GetFactor(nodeCount); // largest factor of number of nodes used for kernel block size
    int backupFactor = tree::GetFactor(nodeCount - 1); // backup largest factor of (number of nodes - 1) used if the number of nodes is prime or has small factors
    std::cout << "Factor: " << factor << ", backup factor: " << backupFactor << "\n";

    int i = 0;
    int valueCount = 0; // number of times searchValue is found in tree

    auto startTimeOverhead = std::chrono::high_resolution_clock::now(); // set start time for performance measurement with overhead
    int *d_searchValue, *d_valueCount, *d_dataArray; // GPU counterparts of variables

    // GPU memory allocations for variables
    try
    {
        cudaMalloc((void **)&d_searchValue, sizeof(int));
        cudaMalloc((void **)&d_valueCount, sizeof(int));
        cudaMalloc((void **)&d_dataArray, nodeCount * sizeof(int));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Could not allocate memory for GPU variables.\n" << e.what() << "\n";
        exit(EXIT_FAILURE);
    }

    // copy value of variables from main memory to GPU memory
    cudaMemcpy(d_searchValue, &searchValue, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valueCount, &valueCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataArray, dataArray, nodeCount * sizeof(int), cudaMemcpyHostToDevice);

    auto startTime = std::chrono::high_resolution_clock::now(); // set start time for performance measurement without overhead
    // choose the bigger factor for best performance
    if(factor >= backupFactor)
    {
        for(; i < nodeCount; i += factor) // for loop jumps by factor every iteration
        {
            FIND<<<1, factor>>>(d_searchValue, d_valueCount, d_dataArray + i); // call kernel with factor threads
        }
    }
    else
    {
        for(; i < nodeCount - 1; i += backupFactor) // for loop jumps by backupFactor every iteration
        {
            FIND<<<1, backupFactor>>> (d_searchValue, d_valueCount, d_dataArray + i);  // call kernel with backupFactor threads
        }
        FIND<<<1, 1>>> (d_searchValue, d_valueCount, d_dataArray + nodeCount); // check last node's data
    }
    cudaDeviceSynchronize(); // wait for all kernels to finish
    auto endTime = std::chrono::high_resolution_clock::now(); // set end time for performance measurement without overhead

    cudaMemcpy(&valueCount, d_valueCount, sizeof(int), cudaMemcpyDeviceToHost); // copy valueCount from GPU to main memory
    auto endTimeOverhead = std::chrono::high_resolution_clock::now(); // set end time for performance measurement with overhead

    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime); // calculate elapsed time in milliseconds without overhead
    auto elapsedTimeOverhead = std::chrono::duration_cast<std::chrono::microseconds>(endTimeOverhead - startTimeOverhead); // calculate elapsed time in milliseconds with overhead
    std::cout << searchValue << " was found " << valueCount << " times.\n";
    std::cout << "Time(ms): " << elapsedTime.count() / 1000.0 << ". Time(ms) with overhead: " << elapsedTimeOverhead.count() / 1000.0 << "\n";

    // free variables
    free(dataArray);
    tree::DestroyTree(root);
    cudaFree(d_valueCount);
    cudaFree(d_dataArray);
    cudaFree(d_searchValue);
    return 0;
}

// kernel to find occurences of searchValue in dataArray
__global__ void FIND(int *d_searchValue, int *d_valueCount, int *d_dataArray)
{
    if(*(d_dataArray+(int)threadIdx.x) == *d_searchValue) // parse dataArray using threadId and match with searchValue
    {
        atomicAdd(d_valueCount, 1); // increment valueCount atomically to avoid memory overwriting issues
    }
}