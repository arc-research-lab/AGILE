// graph-gen: Generate synthetic graphs and store as binary CSR files.
//
// Usage:
//   graph-gen -g 20 -o ./my-graph          # Kronecker graph, 2^20 vertices, unweighted
//   graph-gen -u 20 -o ./my-graph          # Uniform random graph, 2^20 vertices
//   graph-gen -g 20 -o ./my-graph -w       # Kronecker graph with random weights
//   graph-gen -g 20 -o ./my-graph -k 32    # Average degree 32
//
// Output files:
//   <prefix>.offsets.bin    - CSR row offsets (SGOffset per vertex + 1)
//   <prefix>.neighbors.bin  - CSR column indices (NodeID per edge)
//   <prefix>.weights.bin    - edge weights (only when -w is specified)
//   <prefix>.info.txt       - metadata (num_nodes, num_edges, etc.)

#include <iostream>
#include <string>

#include "benchmark.h"
#include "graph_dump.h"


int main(int argc, char* argv[]) {
  // ── Check for our custom flags before passing to gapbs ──
  //    -w / --weights   : generate weighted graph
  //    -o / --output    : output file prefix (required)
  //
  // We strip these from argv before handing off to CLBase so
  // gapbs doesn't choke on unknown flags.

  bool weighted = false;
  std::string output_prefix;

  // Build a new argv without our custom flags.
  std::vector<char*> filtered_argv;
  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-w" || arg == "--weights") {
      weighted = true;
    } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      output_prefix = argv[i + 1];
      ++i;  // skip the value
    } else {
      filtered_argv.push_back(argv[i]);
    }
  }

  if (output_prefix.empty()) {
    std::cerr << "Error: output prefix is required (-o <prefix>)" << std::endl;
    std::cerr << "Usage: graph-gen {-g <scale> | -u <scale>} -o <prefix> [-w] [-k <degree>]" << std::endl;
    return 1;
  }

  int new_argc = static_cast<int>(filtered_argv.size());
  char** new_argv = filtered_argv.data();

  if (weighted) {
    // ── Weighted graph ──
    CLBase cli(new_argc, new_argv, "graph-gen (weighted)");
    if (!cli.ParseArgs()) return 1;

    WeightedBuilder builder(cli);
    WGraph g = builder.MakeGraph();

    std::cout << "Generated weighted graph:" << std::endl;
    std::cout << "  Nodes: " << g.num_nodes() << std::endl;
    std::cout << "  Edges: " << g.num_edges() << " (directed: "
              << g.num_edges_directed() << ")" << std::endl;

    DumpGraphCSR(g, output_prefix);

    std::cout << "Dumped to: " << output_prefix << ".{offsets,neighbors,weights,info}.bin" << std::endl;

  } else {
    // ── Unweighted graph ──
    CLBase cli(new_argc, new_argv, "graph-gen (unweighted)");
    if (!cli.ParseArgs()) return 1;

    Builder builder(cli);
    Graph g = builder.MakeGraph();

    std::cout << "Generated unweighted graph:" << std::endl;
    std::cout << "  Nodes: " << g.num_nodes() << std::endl;
    std::cout << "  Edges: " << g.num_edges() << " (directed: "
              << g.num_edges_directed() << ")" << std::endl;

    DumpGraphCSR(g, output_prefix);

    std::cout << "Dumped to: " << output_prefix << ".{offsets,neighbors,info}.bin" << std::endl;
  }

  return 0;
}
