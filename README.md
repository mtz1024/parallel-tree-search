## Summary

This project implements a simple tree-like structure as well as a serial and parallel (using CUDA) method to search through it for a specific value and returns how many times the value was found.

## Explanation

Each node in the tree comprises of an array of children and a pointer to the nodes data. The reasoning behind having a pointer instead of the actual data is so that all data can be stored sequentially inside an array to allow its easy movement from main memory to GPU memory. Only copying the data array to GPU memory instead of the whole tree structure allows for a much less memory impact (~1:8) as GPU memory is generally smaller than main memory.

The tree is created by making a node with a random or fixed number of children based off of whether ```fully-balanced``` is set or not. For each child, the process of creating a node is repeated via recursion until the maximum depth of the tree is reached. When returning from recursion, a random integer is placed in the data array and the data pointer of the child is set to the address of the integer inside the data array.

The serial search is done through a ```for``` loop iterating over the data array. The parallel search is done through a simple kernel that goes through the data array in chunks after it is copied to GPU memory, the kernel is called in a loop allowing for multiple kernel calls to run at a given time. In order to maximise kernel’s block size, the largest factor of the number of nodes comprising the tree that is smaller than 1024 is used. The reason for the 1024 limit is due to CUDA not allowing for more than 1024 threads per block (CUDA compute 7.5). A backup factor of the number of nodes - 1 is calculated in case the number of nodes is prime or in case it has a smaller factor than the number of nodes - 1.

## Performance

Performance was tested on a laptop with an Intel Core i7 9750H running at 2.6Ghz base and 4.5Ghz boost, a Nvidia RTX 2070 running at 1215Mhz base and 1440Mhz boost, and 16GB of DDR4 RAM running at 2666MHz. Two fully balanced trees were used for testing: one that is 19 levels deep with two children per node (1,048,575 nodes) and one that is 9 levels deep with 8 children per node (153,391,689 nodes). The reason for specifically using these is that they offered the best-case scenario for the GPU with a factor of 1023. In order to get the most objective results, the search value and the random seed was the same across runs. The performance was measured using C++’s ```<chrono>``` library. The breakdown can be seen in the table below, the numbers are an average of 10 runs.

| Small Tree | Time with overhead (ms) | Time without overhead (ms) |
| :--------: | :---------------------: | :------------------------: |
|    GPU     |       111.5±5.263       |       2.487±0.01671        |
|    CPU     |            -            |       6.186±0.1242         |

| Large Tree | Time with overhead (ms) | Time without overhead (ms) |
| :--------: | :---------------------: | :------------------------: |
|    GPU     |       458.6±17.90       |        254.7±16.98         |
|    CPU     |            -            |        289.0±2.058         |

## Installation

Make sure the CUDA Toolkit is installed before attempting to build. To build and install, go to where the directory was cloned/extracted and execute the following your prefered terminal emulator:

    mkdir build && cd build
    cmake ..
    make install .
Afterwards head back to the main directory of the package and run ```./treesearch```, this is the command-line interface for the package. Run

    ./treesearch -h
for a list of options that can be used.

## Developer's Notes

This has been a small side project for me and is in no way completely optimised. There are still improvements that can be made, but I wanted to put it out there and hopefully get some responses from the community on what features to add and what improvements to make.
