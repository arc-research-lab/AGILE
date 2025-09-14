class Configs{
public:
    Configs(int argc, char ** argv): argc(argc), argv(argv){
    }

    bool parse(){
        namespace po = boost::program_options;
        po::options_description opts("Options");
        opts.add_options()
            ("help,h", "Show help")
            ("transfer_size,s", po::value<uint32_t>(&transfer_size)->default_value(4096), "Payload in each DMA command")
            ("monitor_threads,m", po::value<uint32_t>(&monitor_threads)->default_value(1), "Monitor threads")
            ("worker_threads,w", po::value<uint32_t>(&worker_threads)->default_value(1), "Worker threads")
            ("queue_depth,d",  po::value<uint32_t>(&queue_depth)->default_value(128), "DMA queue depth")
            ("queue_num,n",  po::value<uint32_t>(&queue_num)->default_value(1), "DMA queue depth")
            ("command_num,c",  po::value<uint32_t>(&command_num)->default_value(1), "DMA request number")
            ("repeat,r",  po::value<uint32_t>(&repeat)->default_value(1), "repeat times");

        po::positional_options_description pos;
        pos.add("file", -1); // all remaining args → files
        po::variables_map vm;
        try {
            // CLI
            po::store(po::command_line_parser(argc, argv).options(opts).positional(pos).run(), vm);
            if (vm.count("help")) {
                std::cout << "Usage: app [options] [files...]\n" << opts << "\n";
                exit(0);
            }
            po::notify(vm); // checks required(), runs notifiers
        } catch (const po::error& e) {
            std::cerr << "Error: " << e.what() << "\n\n" << opts << "\n";
            return false;
        }
        return true;
    }

    int argc;
    char **argv;

    uint32_t monitor_threads;
    uint32_t worker_threads;
    uint32_t queue_num;
    uint32_t queue_depth;
    uint32_t command_num;
    uint32_t repeat;
    uint32_t transfer_size;

};