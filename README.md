## Summary

The program implements a simple tree-like structure and utilises a CUDA kernel to search through it for a specific value and returns how many times the value was found.

## Explanation

Each node in the tree comprises of an array of children and a pointer to the nodes data. The reasoning behind having a pointer instead of the actual data is so that all data can be stored sequentially inside an array to allow its easy movement from main memory to GPU memory. Only copying the data array to GPU memory instead of the whole tree structure allows for a much less memory impact (~1:8) as GPU memory is generally smaller than main memory.

The tree is created by making a node with a random number of children (If ```fully_balanced = true```, the number of children will be fixed). For each child, the process of creating a node is repeated via recursion until the maximum depth of the tree is reached. When returning from recursion, a random integer is placed in the data array and the data pointer of the child is pointed to the location of the integer inside the data array.

The search is done through a simple kernel that goes through the data array in chunks after it is copied to GPU memory, the kernel is called in a loop meaning that multiple kernels are running at a given time. In order to maximise chunk size, the largest factor of the number of nodes comprising the tree that is smaller than 1024 is used as the chunk size. The reason for the 1024 limit is due to CUDA not allowing for more than 1024 threads per kernel call (CUDA compute 7.5). A backup factor of the number of nodes - 1 is calculated in case the number of nodes is prime or in case it has a smaller factor than the number of nodes - 1.

## Performance

Performance was tested on a laptop with an Intel Core i7 9750H running at 2.6Ghz base and 4.5Ghz boost, a Nvidia RTX 2070 running at 1215Mhz base and 1440Mhz boost, and 16GB of DDR4 RAM running at 2666MHz. Two fully balanced trees were used for testing: one that is 19 levels deep with two children per node (1,048,575 nodes) and one that is 9 levels deep with 8 children per node (153,391,689 nodes). The reason for specifically using these is that they offered the best-case scenario for the GPU with a factor of 1023. The serial implementation is done through a simple ```for``` loop through the data array. Performance breakdown can be seen in the table below, the numbers are an average of 10 runs. 

| Small Tree | Time with overhead (ms) | Time without overhead (ms) |
| :--------: | :---------------------: | :------------------------: |
|    GPU     |      5.034±0.1469       |       3.278±0.04363        |
|    CPU     |            -            |        5.376±1.450         |

| Large Tree | Time with overhead (ms) | Time without overhead (ms) |
| :--------: | :---------------------: | :------------------------: |
|    GPU     |       375.0±16.11       |        272.6±15.93         |
|    CPU     |            -            |        335.1±4.616         |

As can be seen, while in percentage terms the performance is an improvement without overhead (64.00% for the small tree, 22.93% for the large tree), in absolute terms the difference is negligible. This could be due to method inefficiencies or due to the large tree not being big enough for the GPU to gain a better advantage.

## Developer's Notes

This has been a small side project for me and is in no way completely optimised. There are still improvements that can be made (e.g. a function to free the tree), but I wanted to put it out there and hopefully get some responses from the community on what features to add and what improvements to make.