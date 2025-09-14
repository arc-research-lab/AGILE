#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string input;
  int threads;
  bool verbose;
  std::vector<std::string> files;

  po::options_description opts("Options");
  opts.add_options()
    ("help,h", "Show help")
    ("input,i",   po::value<std::string>(&input)->required(), "Input file")
    ("threads,t", po::value<int>(&threads)->default_value(4), "Worker threads")
    ("verbose,v", po::bool_switch(&verbose), "Verbose output")
    ("file",      po::value<std::vector<std::string>>(&files), "Positional files");

  po::positional_options_description pos;
  pos.add("file", -1); // all remaining args → files

  po::variables_map vm;
  try {
    // CLI
    po::store(po::command_line_parser(argc, argv).options(opts).positional(pos).run(), vm);

    // Config file (INI-like): e.g., app.conf containing "threads=8"
    if (std::ifstream conf{"app.conf"}) {
      po::store(po::parse_config_file(conf, opts, true), vm);
    }

    // Environment: APP_THREADS=16 → threads
    po::store(po::parse_environment(opts, [](const std::string& k){
      return k.rfind("APP_", 0) == 0 ? k.substr(4) : std::string{};
    }), vm);

    if (vm.count("help")) {
      std::cout << "Usage: app [options] [files...]\n" << opts << "\n";
      return 0;
    }
    po::notify(vm); // checks required(), runs notifiers
  } catch (const po::error& e) {
    std::cerr << "Error: " << e.what() << "\n\n" << opts << "\n";
    return 1;
  }

  // Use your options...
  std::cout << "input=" << input << ", threads=" << threads
            << ", verbose=" << std::boolalpha << verbose << "\n";
}