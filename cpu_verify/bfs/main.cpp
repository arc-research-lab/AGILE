#include <iostream>
#include <cstring>
#include <fstream>
#include <malloc.h>
#include <queue>
#include <string>
#include <chrono>

#include "config.h"

void bfs(unsigned int * node_array, unsigned int * edge_array, unsigned int * label, 
            unsigned int start_node, unsigned int node_size, unsigned int & visited_num, unsigned int &total_level){
    memset(label, -1, node_size * sizeof(unsigned int));
    int level = 0;
    std::queue<unsigned int> cur_q, next_q;
    cur_q.push(start_node);

    while(!cur_q.empty()){

        while(!cur_q.empty()){
            unsigned int cur_node = cur_q.front();
            cur_q.pop();

            if(label[cur_node] == -1){
                visited_num++;
                label[cur_node] = level;
                for(int i = node_array[cur_node]; i < node_array[cur_node + 1]; ++i){
                    next_q.push(edge_array[i]);
                }
            }
        }

        level++;
        while(!next_q.empty()){
            cur_q.push(next_q.front());
            next_q.pop();
        }

    }
    total_level = level;

}

int main(int argc, char** argv){

    Configs config(argc, argv);
    config.parse();
    config.show_settings();

    unsigned int * node_array = (unsigned int *) malloc(sizeof(unsigned int) * (config.node_num + 1));
    unsigned int * edge_array = (unsigned int *) malloc(sizeof(unsigned int) * config.edge_num);
    unsigned int * label_array = (unsigned int *) malloc(sizeof(unsigned int) * config.node_num);

    std::ifstream node_file(config.node_file, std::ios::binary);
    std::ifstream edge_file(config.edge_file, std::ios::binary);
    std::cout << "load nodes\n";
    unsigned int data;
    for(unsigned int i = 0; i < config.node_num + 1; ++i){
        node_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
        node_array[i] = data;
    }
    
    std::cout << "load edges\n";
    for(unsigned int i = 0; i < config.edge_num; ++i){
        edge_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
        edge_array[i] = data;
    }
    
    int start = config.start_node;
    std::cout << "start bfs @ node " << start << std::endl;

    unsigned int visited_num;
    unsigned int total_level;

    std::chrono::high_resolution_clock::time_point s0, e0;
    s0 = std::chrono::high_resolution_clock::now();

    bfs(node_array, edge_array, label_array, start, config.node_num, visited_num, total_level);

    e0 = std::chrono::high_resolution_clock::now();
    double totaltime = std::chrono::duration_cast<std::chrono::nanoseconds>(e0 - s0).count();

    std::cout << "bfs finished in " << totaltime << " ns\n";
    std::cout << "visited num: " << visited_num << "\n";
    std::cout << "level num: " << total_level << "\n";

    std::ofstream bfs_res(config.output_file, std::ios::out | std::ios::binary);
    for(int i = 0; i < config.node_num; ++i){
        bfs_res.write(reinterpret_cast<const char*>(label_array + i), sizeof(unsigned int));
    }
    bfs_res.close();
    std::cout << "res saved to " << config.output_file << std::endl;

    free(node_array);
    free(edge_array);
    free(label_array);

    return 0;
}