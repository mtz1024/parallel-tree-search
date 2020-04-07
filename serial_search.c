#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

int levels = 9; // Level depth of the tree
int max_children = 8; // Number of maximum children each node can have
int node_count = 0; // Number of nodes inside the tree
int max_node_value = 1999; // Maximum value stored inside tree nodes
bool fully_balanced = true; // Boolean for whether to create a random or fully balanced tree

// Structure for tree node
struct node {
    int *data; // Pointer to node data
    struct node *children[]; // Array of pointers to node children
};

// Prototypes
struct node *constructTree(int *, int);

int main(void) {
    srand((unsigned)time(NULL)); // Reset random seed

    int l;
    long int max_nodes = 0; // Number of maximum nodes that can be created for the tree
    for(l = 0; l <= levels; ++l) {
        max_nodes += pow(max_children, l);
    }
    int *data_array = (int *)malloc(max_nodes * sizeof(int)); // Allocate memory for nodes data based off of maximum nodes as repeated calls for realloc is time consuming

    struct node *root; // Root node of tree
    root = constructTree(data_array, 0); // Constuct tree
    printf("Tree created with %d nodes\n", node_count);

    int value; // Value to search for inside tree
    printf("Enter search value: ");
    scanf("%d", &value);

    int i;
    clock_t start_time, end_time; // Timers for performace
    start_time = clock(); // set start time
    int value_count = 0; // Number of times the requested value is found in the tree
    for(i = 0; i < node_count; ++i) { // Go through array and check if values equal requested value
        if(*(data_array + i) == value) {
            ++value_count;
        }
    }
    end_time = clock(); // set end time
    double elapsed_time = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000; // Calculate elapsed time as milliseconds
    printf("%d was found %d times\n", value, value_count);
    printf("Time: %lf ms\n", elapsed_time);

    // Free variables
    free(data_array);
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