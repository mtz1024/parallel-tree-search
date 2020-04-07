#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cuda.h>
#include <time.h>

int levels = 9; // Level depth of the tree
int max_children = 8; // Number of maximum children each node can have
int node_count = 0; // Number of nodes inside the tree
int max_node_value = 1999; // Maximum value stored inside tree nodes
bool fully_balanced = true; // Boolean for whether to create a random or fully balanced tree

// Structure for tree node
struct node {
    struct node *children[]; // Array of pointers to node children
    int *data; // Pointer to node data
};

// Prototypes
struct node *constructTree(int *, int);
int getFactor(int);
__global__ void FIND(int *, int *, int *);

int main(void) {
    srand((unsigned)time(NULL)); // Reset random seed

    long int max_nodes = 0; // Number of maximum nodes that can be created for the tree
    int l;
    for(l = 0; l <= levels; ++l) { // Calculate number of max nodes
        max_nodes += pow(max_children, l);
    }
    int *data_array = (int *)malloc(max_nodes * sizeof(int)); // Allocate memory for nodes data based off of maximum nodes as repeated calls for realloc is time consuming

    struct node *root; // Root node of tree
    root = constructTree(data_array, 0); // Constuct tree
    printf("Tree created with %d nodes\n", node_count);

    int value; // Value to search for inside tree
    printf("Enter search value: ");
    scanf("%d", &value);

    int factor; // Largest factor of number of nodes used for size of batches to process on GPU
    int backup_factor; // Backup largest factor of (number of nodes - 1) in case the number of nodes is prime or has small factors
    factor = getFactor(node_count);
    backup_factor = getFactor(node_count - 1);

    int i;
    int value_count = 0; // Number of times the requested value is found in the tree

    cudaEvent_t start_time, end_time; // Timers without overhead
    cudaEvent_t start_time_overhead, end_time_overhead; // timers with overhead

    // Set timers
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    cudaEventCreate(&start_time_overhead);
    cudaEventCreate(&end_time_overhead);

    cudaEventRecord(start_time_overhead, 0); // Start timer with overhead
    int *d_value, *d_value_count, *d_data_array; // GPU couterparts of variables

    // GPU memory allocations for variables
    cudaMalloc((void **)&d_value, sizeof(int));
    cudaMalloc((void **)&d_value_count, sizeof(int));
    cudaMalloc((void **)&d_data_array, node_count * sizeof(int));

    // Copy value of variables from main memory to GPU memory
    cudaMemcpy(d_value, &value, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_count, &value_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_array, data_array, node_count * sizeof(int), cudaMemcpyHostToDevice);

    // Choose the bigger factor for best performance
    if(factor > backup_factor) {
        printf("Factor: %d\n", factor);
        cudaEventRecord(start_time, 0); // Start timer without overhead

        for(i = 0; i < node_count; i += factor) { // For jumps by factor every loop
            FIND<<<1, factor>>>(d_value, d_value_count, d_data_array + i); // Call kernel with factor threads
        }
    }
    else {
        printf("Backup factor: %d\n", backup_factor);
        cudaEventRecord(start_time, 0); // Start timer without overhead
        for(i = 0; i < node_count - 1; i += backup_factor) { // For jumps by backup_factor every loop
            FIND<<<1, backup_factor>>> (d_value, d_value_count, d_data_array + i);  // Call kernel with backup_factor threads
        }
        FIND<<<1, 1>>> (d_value, d_value_count, d_data_array + node_count); // Check last node's data
    }
    cudaDeviceSynchronize(); // Wait for all kernels to finish

    cudaEventRecord(end_time, 0); // End timer without overhead
    cudaEventSynchronize(end_time); // Syncronise timer without overhead

    cudaMemcpy(&value_count, d_value_count, sizeof(int), cudaMemcpyDeviceToHost); // Copy count from GPU to main memory

    cudaEventRecord(end_time_overhead, 0); // End timer with overhead
    cudaEventSynchronize(end_time_overhead); // Syncronise timer with overhead

    float elapsed_time;
    float elapsed_time_overhead;
    cudaEventElapsedTime(&elapsed_time, start_time, end_time); // Set elapsed time without overhead
    cudaEventElapsedTime(&elapsed_time_overhead, start_time_overhead, end_time_overhead); // Set elapsed time with overhead

    printf("%d was found %d times\n", value, value_count);
    printf("Time: %f ms\n", elapsed_time);
    printf("Time with overhead: %f ms\n", elapsed_time_overhead);

    // Destroy timers
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
    cudaEventDestroy(start_time_overhead);
    cudaEventDestroy(end_time_overhead);

    // Free variables
    free(data_array);
    cudaFree(d_value_count);
    cudaFree(d_data_array);
    cudaFree(d_value);
    return 0;
}

// Function to create tree
struct node *constructTree(int *data_array, int level) {
    // Construct nodes until we reach maximum depth of tree, then return NULL
    if(level <= levels) {
        int *next_level = (int *)malloc(sizeof(int)); // Variable of next tree level for each node based off of passed level variable
        *next_level = level + 1;

        int *children_count = (int *)malloc(sizeof(int)); // Number of node children
        if(fully_balanced) {
            *children_count = max_children; // Always give maximum number of children for a fully-balanced tree
        }
        else {
            *children_count = rand() % (max_children + 1); // Give random number of children for a non-fully-balanced tree
        }
        struct node *temp = (struct node *)malloc(sizeof(struct node) + (*children_count * sizeof(struct node *))); // Allocate memory for node based off of number of children

        int i;
        for(i = 0; i < *children_count; ++i) {
            temp->children[i] = constructTree(data_array, *next_level); // Construct nodes of children
            // Check if we are at the end of tree
            if(temp->children[i] != NULL) {
                data_array[node_count] = rand() % (max_node_value + 1); // Set node data value to a random int between zero and max_node_value
                temp->children[i]->data = &(data_array[node_count++]); // Set data pointer inside child node to same location as inside data_array
            }
        }
        // Special case for root node
        if(level == 0) {
            data_array[node_count] = rand() % (max_node_value + 1);
            temp->data = &(data_array[node_count++]);
        }
        // Free variables
        free(children_count);
        free(next_level);

        return temp; // Return created node
    }
    else {
        return NULL;
    }
}

// Function to find largest factor
int getFactor(int node_count) {
    int factor, i;
    for(i = 1; i < 1024; ++i) { // Find a factor smaller than 1024 as that is the maximum number of threads per kernel call (based off of cuda compute 7.5)
        if(node_count % i == 0) {
            factor = i;
        }
    }
    return factor;
}

// Kernel to find requested value in array
__global__ void FIND(int *d_value, int *d_value_count, int *d_data_array) {
    // Check value based off of thread id
    if(*(d_data_array+(int)threadIdx.x) == *d_value) {
        *d_value_count= *d_value_count + 1;
    }
}