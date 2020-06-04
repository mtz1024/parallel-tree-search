#pragma once

#include <iostream>

namespace tree
{
    struct Node // tree node
    {
        int *data; // pointer to node data
        int childCount; // number of children the node has
        struct Node *children[]; // array of pointers to node children
    };
    /**
     * generate tree and return its head
     * @param dataArray: the head of the data array of the tree
     * @param levels: level depth of the tree
     * @param maxChildren: maximum number of children a node can have
     * @param maxNodeValue: maximum data value that can be stored in a node
     * @param fullyBalanced: whether to generate a random or fully balanced tree
     * @param currentLevel: the current level reached in tree
     * @return head of generated tree
     */
    struct Node *GenerateTree(int *dataArray, int *levels, int *maxChildren, int *maxNodeValue, bool *fullyBalanced, int currentLevel=0);

    /**
     * destroy tree by freeing it from memory
     * @param node: head of tree
     */
    void DestroyTree(struct Node *node);

    /**
     * get largest factor of passed number that is less than 1024
     * @param number: the number to find the factor for
     * @return factor of number
     */
    int GetFactor(int number);

}
extern int nodeCount; // number of nodes inside the tree