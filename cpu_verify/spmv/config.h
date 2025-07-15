

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
   
    unsigned int node_num;
    unsigned long edge_num;
    unsigned int start_node;
    unsigned int max_itr;
    float error_thresh;
    std::string node_file;
    std::string edge_file;
    std::string output_file;
    std::string output_norm_val;

    Configs(int argc, char ** argv){
        this->argc = argc;
        this->argv = argv;

        // Application
        items.push_back(new ConfigItem<unsigned int>("node_num", "-nn", 4847571, "Total node count of the graph", node_num));
        items.push_back(new ConfigItem<unsigned long>("edge_num", "-en", 68993773, "Total edge count of the graph", edge_num));
        items.push_back(new ConfigItem<unsigned int>("start_node", "-sn", 0, "Start node index", start_node));
        items.push_back(new ConfigItem<unsigned int>("max_itr", "-mi", 1, "The max iteration times", max_itr));
        items.push_back(new ConfigItem<float>("error_thresh", "-e", 0, "The stop threshold", error_thresh));
        items.push_back(new ConfigItem<std::string>("node_file", "-nf", "./datasets/convert/row.bin", "Path of the indptr file", node_file));
        items.push_back(new ConfigItem<std::string>("edge_file", "-ef", "./datasets/convert/col.bin", "Path of the indices file", edge_file));
        items.push_back(new ConfigItem<std::string>("output_file", "-o", "cpu-res-pr.txt", "Output file", output_file));
        items.push_back(new ConfigItem<std::string>("output_norm_val", "-onv", "pr-vals.bin", "Output file", output_norm_val));

        this->config_size = this->items.size();
        check_config_name();
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
                    } else if (this->items[j]->typeName.compare("unsigned long") == 0){
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