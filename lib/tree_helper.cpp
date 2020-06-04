#include "tree.hpp"

int nodeCount = 0; // number of nodes inside the tree

// function to generate tree and return its head
struct tree::Node *tree::GenerateTree(int *dataArray, int *levels, int *maxChildren, int *maxNodeValue, bool *fullyBalanced, int currentLevel)
{
    // construct nodes until we reach maximum depth of tree, then return NULL
    if(currentLevel <= *levels)
    {
        int childrenCount; // number of node children
        if(*fullyBalanced)
        {
            childrenCount = *maxChildren; // always create maximum number of children in a fully-balanced tree
        }
        else
        {
            childrenCount = std::rand() % (*maxChildren + 1); // create random number of children in a random tree
        }
        struct tree::Node *node; // initialize current node
        try
        {
            node = (struct tree::Node *)malloc(sizeof(struct tree::Node) + (childrenCount * sizeof(struct tree::Node *))); // allocate memory for node based off of number of children
        }
        catch(const std::exception& e)
        {
            std::cerr << "Could not allocate memory for tree node.\n" << e.what() << "\n";
            exit(EXIT_FAILURE);
        }
        node->childCount = childrenCount;
        int i;
        for(i = 0; i < childrenCount; ++i)
        {
            node->children[i] = GenerateTree(dataArray, levels, maxChildren, maxNodeValue, fullyBalanced, currentLevel + 1); // Construct nodes of children
            // check if we are at the end of tree
            if(node->children[i] != NULL)
            {
                dataArray[nodeCount] = std::rand() % (*maxNodeValue + 1); // set node data to a random integer between zero and maxNodeValue
                node->children[i]->data = &(dataArray[nodeCount++]); // set data pointer inside child node to data memory address inside dataArray
            }
            else
            {
                node->childCount = 0; // set childCount to zero if we are at the leaves
            }
            
        }
        // special case for root node
        if(currentLevel == 0)
        {
            dataArray[nodeCount] = std::rand() % (*maxNodeValue + 1);
            node->data = &(dataArray[nodeCount++]);
        }
        return node; // return created node
    }
    else
    {
        return NULL;
    }
}

// function to destroy tree by freeing it from memory
void tree::DestroyTree(struct tree::Node *node)
{
    // if nodeCount is not zero keep parsing tree else we are at the leaves
    if(node->childCount)
    {
        int i = 0;
        for(; i < node->childCount; ++i)
        {
            tree::DestroyTree(node->children[i]);
        }
    }
    else
    {
        free(node);
    }
    
}

// function to find and return largest factor of passed number that is less than 1024
int tree::GetFactor(int number)
{
    int factor, i;
    // find a factor smaller than 1024 as that is the maximum number of threads allowed per block (based off of cuda compute 7.5)
    for(i = 1; i < 1024; ++i)
    {
        factor = number % i == 0 ? i : factor;
    }
    return factor;
}