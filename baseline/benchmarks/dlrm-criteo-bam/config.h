

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <cxxabi.h>
#include <utility>
#include <sstream>
#include <vector>

template<typename T>
std::string type_name()
{
    int status;
    std::string tname = typeid(T).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if(status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
    }   
    return tname;
}

class ConfigBase{
public:
    std::string configName;
    std::string configAbbr;
    std::string desc;
    std::string typeName;

    virtual std::string getType() = 0;
    virtual std::string getDefualt() = 0;
};

template<typename T>
T parse_from_str(std::string str);

template<>
std::string parse_from_str(std::string str){
    return str;
}

template<typename T>
T parse_from_str(std::string str){
    std::stringstream convert(str);
    T val;
    convert >> val;
    return val;
}


template<typename T>
class ConfigItem: public ConfigBase{
    
    T defualt_val;
    
public:
    T *val;

    ConfigItem(std::string configName, std::string configAbbr, T defualt_val, std::string desc, T & val){
        this->configName = configName;
        this->configAbbr = configAbbr;
        this->defualt_val = defualt_val;
        this->val = &val;

        *(this->val) = defualt_val;
        // val = defualt_val;
        this->desc = desc;
        
        if(typeid(T).name() == typeid(std::string).name()){
            this->typeName = "string";
        } else {
            this->typeName = type_name<T>();
        }
    }

    std::string getType() override {
        return typeName; 
    }

    std::string getDefualt(){
        std::ostringstream ss;
        ss << defualt_val;
        return ss.str();
    }
    void parse(std::string str){
        *(this->val) = parse_from_str<T>(str);
    }

};


class Configs{
    int argc;
    char ** argv;
    std::vector<ConfigBase *> items;
    unsigned int config_size;

public:
    // BaM config
    unsigned int slot_size;
    unsigned int gpu_slot_num;
    unsigned int cudaDevice;
    unsigned int n_ctrls;
    unsigned int memalloc;

    unsigned int ssdtype;
    unsigned int nvmNamespace;
    unsigned long numElems;

    // NVME config
    std::string nvme_bar;
    unsigned int bar_size;
    unsigned int ssd_blk_offset;
    unsigned int queue_num;
    unsigned int queue_depth;

    // Parallelism config
    unsigned int block_dim;
    unsigned int thread_dim;
    unsigned int agile_dim;

    // Application config
    unsigned int iteration;
    unsigned int ssd_block_num;
    unsigned int enable_compute;
    unsigned int enable_load;

    unsigned int embedding_vec_dim;
    unsigned int batch_size;
    unsigned int dense_input_dim;
    unsigned int sparse_input_dim;

    std::string output_file;
    std::string bottom_layer;
    std::string top_layer;
    // std::string project_layer;

    std::string weights_file;
    std::string dense_input_file;
    std::string sparse_input_file;

    Configs(int argc, char ** argv){
        this->argc = argc;
        this->argv = argv;
        // BaM config
        items.push_back(new ConfigItem<unsigned int>("slot_size", "-ss", 4096, "Slot size of the cache", slot_size));
        items.push_back(new ConfigItem<unsigned int>("gpu_slot_num", "-gsn", 65536 * 8, "Number of slots in the gpu cache", gpu_slot_num));
        items.push_back(new ConfigItem<unsigned int>("cudaDevice", "-g", 0, "CUDA device id", cudaDevice));
        items.push_back(new ConfigItem<unsigned int>("n_ctrls", "-k", 1, "Number of NVMe controllers", n_ctrls));
        items.push_back(new ConfigItem<unsigned int>("memalloc", "-m", 2, "Memory allocation type", memalloc));

        items.push_back(new ConfigItem<unsigned int>("ssdtype", "-s", 0, "SSD type", ssdtype));
        items.push_back(new ConfigItem<unsigned int>("nvmNamespace", "-n", 1, "NVMe namespace", nvmNamespace));
        items.push_back(new ConfigItem<unsigned long>("numElems", "-ne", 8589934592, "Number of elements in the backing array", numElems));

        // NVME config
        items.push_back(new ConfigItem<std::string>("nvme_bar", "-bar", "0x97000000", "PCIe bar address of target ssd", nvme_bar));
        items.push_back(new ConfigItem<unsigned int>("bar_size", "-bar_size", 32768, "PCIe bar size of target ssd", bar_size));
        items.push_back(new ConfigItem<unsigned int>("queue_num", "-qn", 32, "Number of NVME queue pairs", queue_num));
        items.push_back(new ConfigItem<unsigned int>("queue_depth", "-qd", 256, "Depth of each NVME queue", queue_depth));
        items.push_back(new ConfigItem<unsigned int>("ssd_blk_offset", "-bo", 0, "Offset of ssd blocks", ssd_blk_offset));

        // parallelism config
        items.push_back(new ConfigItem<unsigned int>("block_dim", "-bd", 32, "Block dimension", block_dim));
        items.push_back(new ConfigItem<unsigned int>("thread_dim", "-td", 1024, "Thread dimension", thread_dim));
        items.push_back(new ConfigItem<unsigned int>("agile_dim", "-ad", 1, "Agile dimension", agile_dim));

        // application config
        items.push_back(new ConfigItem<std::string>("output_file", "-o", "dlrm_embedding_vec_sum.bin", "output file", output_file));
        items.push_back(new ConfigItem<unsigned int>("iteration", "-itr", 100, "number of iterations", iteration));
        items.push_back(new ConfigItem<unsigned int>("ssd_block_num", "-sbn", 65536 * 8, "number of ssd blocks", ssd_block_num));
        items.push_back(new ConfigItem<unsigned int>("enable_compute", "-ec", 1, "enable compute", enable_compute));
        items.push_back(new ConfigItem<unsigned int>("enable_load", "-el", 1, "enable load", enable_load));
        items.push_back(new ConfigItem<unsigned int>("embedding_vec_dim", "-evd", 64, "embedding dimension", embedding_vec_dim));
        items.push_back(new ConfigItem<unsigned int>("dense_input_dim", "-did", 13, "dense input dimension", dense_input_dim));
        items.push_back(new ConfigItem<unsigned int>("sparse_input_dim", "-sid", 26, "sparse input dimension", sparse_input_dim));
        items.push_back(new ConfigItem<unsigned int>("batch_size", "-bs", 128, "batch size", batch_size));

        items.push_back(new ConfigItem<std::string>("bottom_layer", "-bl", "512-512-512", "bottom layers", bottom_layer));
        // items.push_back(new ConfigItem<std::string>("project_layer", "-pl", "1024", "projection layer", project_layer));
        items.push_back(new ConfigItem<std::string>("top_layer", "-tl", "1024-1024-1024", "top layers", top_layer));
        items.push_back(new ConfigItem<std::string>("weights_file", "-wf", "/home/zhuoping/workspace/datasets/dlrm_weights.bin", "weights file", weights_file));
        items.push_back(new ConfigItem<std::string>("dense_input_file", "-dif", "/home/zhuoping/workspace/datasets/dense.bin", "dense input file", dense_input_file));
        items.push_back(new ConfigItem<std::string>("sparse_input_file", "-sif", "/home/zhuoping/workspace/datasets/categorical.bin", "sparse input file", sparse_input_file));

        this->config_size = this->items.size();
        check_config_name();
        parse();
    }

    ~Configs(){
        for(int i = 0; i < this->config_size; ++i){
            delete this->items[i];
        }
    }

    void check_config_name(){
        for(int i = 0; i < this->config_size; ++i){
            for(int j = i + 1; j < this->config_size; ++j){
                if(this->items[i]->configAbbr.compare(this->items[j]->configAbbr) == 0){
                    std::cout << "duplicate configAbbr: " << this->items[i]->configAbbr << std::endl;
                    exit(0);
                }
            }
        }
    }

    void parse(){
        for(int i = 1; i < this->argc; i += 2){
            bool find = false;
            
            std::string flag(this->argv[i]);

            if(flag.compare("-h") == 0){
                this->help();
                exit(0);
            }

            for(int j = 0; j < this->config_size; ++j){
                if(this->items[j]->configAbbr.compare(flag) == 0){
                    find = true;
                    if(this->items[j]->typeName.compare("unsigned int") == 0){
                        ((ConfigItem<unsigned int>*) (this->items[j]))->parse(this->argv[i + 1]);
                    } else if(this->items[j]->typeName.compare("unsigned long") == 0){
                        ((ConfigItem<unsigned long>*) (this->items[j]))->parse(this->argv[i + 1]);
                    } else if(this->items[j]->typeName.compare("string") == 0){
                        ((ConfigItem<std::string>*) (this->items[j]))->parse(this->argv[i + 1]);
                    } else if(this->items[j]->typeName.compare("float") == 0){
                        ((ConfigItem<float>*) (this->items[j]))->parse(this->argv[i + 1]);
                    } else if (this->items[j]->typeName.compare("bool") == 0){
                        ((ConfigItem<bool>*) (this->items[j]))->parse(this->argv[i + 1]);
                    }
                }
            }

            if(!find){
                std::cout << "option not found: " << this->argv[i] << std::endl;
                std::cout << "help: " << std::endl;
                this->help();
                exit(0);
            }
        }
    }

    void help(){
        std::cout << std::setw(20) << "Config Name" << std::setw(10) << "Option" << std::setw(20) << "type" << std::setw(15) << "defualt" << "\tDescription" << std::endl;
        for(int i = 0; i < this->config_size; ++i){
            // std::cout << this->items[i]->getType() << std::endl;
            std::cout << std::setw(20) << this->items[i]->configName << std::setw(10) << this->items[i]->configAbbr << std::setw(20) << this->items[i]->getType() << std::setw(15) << this->items[i]->getDefualt() << "\t" << this->items[i]->desc << std::endl;
        }
    }

    void show_settings(){
        std::cout << "---------------Config---------------\n";
        for(int i = 0; i < this->config_size; ++i){
            if(this->items[i]->typeName.compare("unsigned int") == 0){
                unsigned int val = *((ConfigItem<unsigned int>*) (this->items[i]))->val;
                std::cout << this->items[i]->configName << ": " << val << std::endl;
            } else if(this->items[i]->typeName.compare("unsigned long") == 0){
                unsigned int val = *((ConfigItem<unsigned long>*) (this->items[i]))->val;
                std::cout << this->items[i]->configName << ": " << val << std::endl;
            } else if(this->items[i]->typeName.compare("string") == 0){
                std::string val = *((ConfigItem<std::string>*) (this->items[i]))->val;
                std::cout << this->items[i]->configName << ": " << val << std::endl;
            } else if(this->items[i]->typeName.compare("float") == 0){
                float val = *((ConfigItem<float>*) (this->items[i]))->val;
                std::cout << this->items[i]->configName << ": " << val << std::endl;
            } else if(this->items[i]->typeName.compare("bool") == 0){
                bool val = *((ConfigItem<bool>*) (this->items[i]))->val;
                std::cout << this->items[i]->configName << ": " << val << std::endl;
            }
        }
        std::cout << "-------------------------------------\n";
    }

};