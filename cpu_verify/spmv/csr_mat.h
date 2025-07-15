#include <memory.h>
#include <iostream>
#include <math.h>

void normlize_row(const unsigned int * offsets, const unsigned int * indices, float * value_arr, unsigned int nodes){
    for(int i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        unsigned int node_count = end - start;
        float node_val = 1.0 / node_count;
        for(int j = start; j < end; ++j){
            value_arr[j] = node_val;
        }
    }
}

void normlize_col(const unsigned int * offsets, const unsigned int * indices, float * value_arr, unsigned int nodes){
    
    // step 0 obtain column count
    unsigned int * column_count = (unsigned int *) malloc(sizeof(unsigned int) * nodes);
    memset(column_count, 0, sizeof(unsigned int) * nodes);

    for(unsigned i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(unsigned int j = start; j < end; ++j){
            unsigned int col = indices[j];
            column_count[col]++;
        }
    }
    // step 1 normalize column
    for(unsigned int i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(unsigned int j = start; j < end; ++j){
            unsigned int col = indices[j];
            float val = 1.0 / column_count[col];
            value_arr[j] = val;
        }
    }
}


void transpose_csr(const unsigned int * offsets, const unsigned int * indices, const float * value_arr, const unsigned int nodes, unsigned int * offsets_t, unsigned int * indices_t, float * value_arr_t){
    // step 0 count nodes in each column, 
    unsigned int * column_count = (unsigned int *) malloc(sizeof(unsigned int) * (nodes));
    memset(column_count, 0, sizeof(unsigned int) * (nodes));
    for(int i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(int j = start; j < end; ++j){
            unsigned int col = indices[j];
            column_count[col] += 1;
        }
    }

    // for(int i = 0; i < nodes; ++i){
    //     std::cout << "col: " << i << " count: " << column_count[i] << std::endl;
    // }

    // std::cout << "step 0\n";
    // step 1 calculate offset and generate offset start arrat
    unsigned int * new_offset = (unsigned int *) malloc(sizeof(unsigned int) * (nodes));
    unsigned int offset_val = 0;
    offsets_t[0] = offset_val;
    for(int i = 0; i < nodes; ++i){
        new_offset[i] = offset_val;
        offset_val += column_count[i];
        offsets_t[i + 1] = offset_val;
        // std::cout << "offset: " << i << " val: " << offsets_t[i + 1] << " offset_val: " << offset_val << std::endl;
    }
    // std::cout << "step 1\n";
    
    // step 2 generate indices and value array
    for(int i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(int j = start; j < end; ++j){
            unsigned int node = indices[j];
            unsigned int offset_t = new_offset[node];
            indices_t[offset_t] = i;
            value_arr_t[offset_t] = value_arr[j];
            new_offset[node]++;
        }
    }
    // std::cout << "step 2\n";
}

void show_csr(const unsigned int * offsets, const unsigned int * indices, const float * value_arr, const unsigned int nodes, const unsigned int edges){
    std::cout << "offsets: [";
    for(int i = 0; i < nodes + 1; ++i){
         std::cout << offsets[i] << "\t";
    }
    std::cout << "]\n";

    std::cout << "indices: [";
    for(int i = 0; i < edges; ++i){
         std::cout << indices[i] << "\t";
    }
    std::cout << "]\n";

    std::cout << "value: [";
    for(int i = 0; i < edges; ++i){
         std::cout << value_arr[i] << "\t";
    }
    std::cout << "]\n";

}

void show_csr_as_dense(const unsigned int * offsets, const unsigned int * indices, const float * value_arr, const unsigned int nodes){
    // float * mat = (float*) malloc(sizeof(float) * nodes * nodes);
    // memset(mat, 0, sizeof(float) * nodes * nodes);
    float mat[4 * 4] = {0};
    for(int i = 0; i < nodes; ++i){
        unsigned int start = offsets[i];
        unsigned int end = offsets[i + 1];
        for(int j = start; j < end; ++j){
            unsigned int col = indices[j];
            mat[i * nodes + col] = value_arr[j];
        }
    }
    std::cout << "Mat:\n";
    for(int i = 0; i < nodes; ++i){
        for(int j = 0; j < nodes; ++j){
            std::cout << mat[i * nodes + j] << "\t";
        }
        std::cout << std::endl;
    }
}

float pr_mat_vec_mul(unsigned int * offsets, unsigned int * indices, float * value_arr, float * vec_arr, float * out_vec_arr, unsigned int nodes, float damping){
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