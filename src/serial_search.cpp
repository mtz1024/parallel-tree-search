#include <cmath>
#include <chrono>

#include "../lib/tree.hpp"

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
    fullyBalanced = argc >= 5 ? std::stoi(argv[4]) : fullyBalanced;

    int l = 0;
    long int maxNodes = 0; // maximum number of nodes that tree can contain
    // calculate maxNodes
    for(; l <= levels; ++l)
    {
        maxNodes += std::pow(maxChildren, l);
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

    int i = 0;
    int valueCount = 0; // number of times searchValue is found in tree
    auto startTime = std::chrono::high_resolution_clock::now(); // set start time for performance measurement
    for(; i < nodeCount; ++i) // parse dataArray and match with searchValue
    {
        if(*(dataArray + i) == searchValue)
        {
            ++valueCount;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now(); // set end time for performance measurement
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime); // calculate elapsed time in milliseconds
    std::cout << searchValue << " was found " << valueCount << " times.\n";
    std::cout << "Time(ms): " << elapsedTime.count() / 1000.0 << "\n";

    // free variables
    free(dataArray);
    tree::DestroyTree(root);
    return 0;
}