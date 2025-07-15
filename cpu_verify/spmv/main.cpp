#include <iostream>
#include <fstream>
#include <math.h>

#include "csr_mat.h"
#include "config.h"

float csr_mat_vec_mul_pr(unsigned int * offsets, unsigned int * indices, float * value_arr, float * vec_arr, float * out_vec_arr, unsigned int nodes, float damping){
    // memset(out_vec_arr, 0, sizeof(float) * nodes);
    for(unsigned int i = 0; i < nodes; ++i){
        out_vec_arr[i] = 0;
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(unsigned int j = start; j < end; ++j){
            out_vec_arr[i] += value_arr[j] * vec_arr[indices[j]];
        }
        out_vec_arr[i] *= damping;
        out_vec_arr[i] += (1.0 - damping) / nodes;
    }
    float norm = 0;
    for(unsigned int i = 0; i < nodes; ++i){
        norm += (out_vec_arr[i] - vec_arr[i]) * (out_vec_arr[i] - vec_arr[i]);
    }
    return sqrt(norm);
}

template<typename T>
void save_data_txt(std::string path, T * arr, unsigned int size){
    std::ofstream distance_bin(path, std::ios::out);
    for(unsigned int i = 0; i < size; ++i){
        // distance_bin.write(reinterpret_cast<const char*>(&arr[i]), sizeof(T));
        distance_bin << arr[i] << "\n";
    }
    distance_bin.close();
}


int main(int argc, char** argv){
    // if(argc < 3){
    //     printf("argv[1]: nodes\n");
    //     printf("argv[2]: edges\n");
    //     return 0;
    // }
    Configs config(argc, argv);
    config.parse();
    config.show_settings();
    unsigned int nodes = config.node_num;
    unsigned int edges = config.edge_num;
    unsigned int * node_array = (unsigned int *) malloc(sizeof(unsigned int) * (nodes + 1));
    unsigned int * edge_array = (unsigned int *) malloc(sizeof(unsigned int) * (edges));
    float * value_arr = (float *) malloc(sizeof(float) * edges);

    unsigned int data;

    std::ifstream node_file(config.node_file, std::ios::binary);
    std::ifstream edge_file(config.edge_file, std::ios::binary);
    std::cout << "load nodes\n";

    for(unsigned int i = 0; i < config.node_num + 1; ++i){
        node_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
        node_array[i] = data;
    }

    std::cout << "load edges\n";
    for(unsigned int i = 0; i < config.edge_num; ++i){
        edge_file.read(reinterpret_cast<char*>(&data), sizeof(unsigned int));
        edge_array[i] = data;
    }
    std::cout << "normlize_col\n";
    // normlize_row(offset, indices, value_arr, nodes);
    normlize_col(node_array, edge_array, value_arr, nodes);

    std::cout << "save val\n";
    std::ofstream norm_val(config.output_norm_val, std::ios::out | std::ios::binary);
    for(unsigned int i = 0; i < config.edge_num; ++i){
        norm_val.write(reinterpret_cast<const char*>(value_arr + i), sizeof(float));
    }
    norm_val.close();

    // unsigned int * offset_t = (unsigned int *) malloc(sizeof(unsigned int) * (nodes + 1));
    // unsigned int * indices_t = (unsigned int *) malloc(sizeof(unsigned int) * (edges));
    // float * value_arr_t = (float *) malloc(sizeof(float) * edges);

    // transpose_csr(offset, indices, value_arr, nodes, offset_t, indices_t, value_arr_t);
    std::cout << "computing\n";
    float * vec = (float *) malloc(sizeof(float) * nodes);
    float * out_vec = (float *) malloc(sizeof(float) * nodes);
    float init_val = 1.0 / nodes;
    for(unsigned int i = 0; i < nodes; ++i){
        vec[i] = init_val;
    }

    unsigned int itr = 0;
    float norm = 0;
    do{
        norm = pr_mat_vec_mul(node_array, edge_array, value_arr, vec, out_vec, nodes, 0.85);
        // norm = pr_mat_vec_mul(offset_t, indices_t, value_arr_t, vec, out_vec, nodes, 0.85);
        memcpy(vec, out_vec, nodes * sizeof(float));
        std::cout << "itr: " << itr << " norm: " << norm << std::endl;
        itr++;
    }while(itr < config.max_itr || norm < config.error_thresh);
    float max_val = 0;
    float min_val = 1;
    for(unsigned int i = 0; i < nodes; ++i){
        if(max_val < vec[i]){
            max_val = vec[i];
        }
        if(min_val > vec[i]){
            min_val = vec[i];
        }
        // if(vec[i] != init_val)
        // std::cout << "node: " << i << " val: " << vec[i] << " ";
    }
    // std::cout << "\nmax:" << max_val << " min: " << min_val << " init: " << init_val << std::endl;

    save_data_txt(config.output_file, vec, nodes);

    

    free(node_array);
    free(edge_array);
    free(value_arr);
   
    return 0;
}