// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_DUMP_H_
#define GRAPH_DUMP_H_

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "graph.h"

// Round up to the next power of two (returns at least 1).
static inline uint64_t next_pow2(uint64_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
  v |= v >> 8;  v |= v >> 16; v |= v >> 32;
  return v + 1;
}

// Pad an already-closed file to the next power-of-two size with zeros.
// Returns the padded file size.
static inline uint64_t pad_file_to_pow2(const std::string &path) {
  std::ifstream in(path, std::ios::ate | std::ios::binary);
  if (!in) return 0;
  uint64_t cur = static_cast<uint64_t>(in.tellg());
  in.close();

  uint64_t target = next_pow2(cur);
  if (target > cur) {
    std::ofstream out(path, std::ios::app | std::ios::binary);
    uint64_t pad_len = target - cur;
    // Write in chunks to avoid huge stack allocs
    const size_t BUF = 4096;
    char zeros[BUF];
    std::memset(zeros, 0, BUF);
    while (pad_len > 0) {
      size_t n = (pad_len < BUF) ? static_cast<size_t>(pad_len) : BUF;
      out.write(zeros, n);
      pad_len -= n;
    }
    out.close();
  }
  return target;
}


template <typename NodeID_, typename DestID_>
class GraphDumper {
 public:
  static void DumpCSR(const CSRGraph<NodeID_, DestID_> &g,
                      const std::string &prefix) {
    if (prefix == "") {
      std::cout << "No output prefix given" << std::endl;
      std::exit(-5);
    }

    std::string offsets_name = prefix + ".offsets.bin";
    std::string neighbors_name = prefix + ".neighbors.bin";

    std::fstream offsets_out(offsets_name, std::ios::out | std::ios::binary);
    if (!offsets_out) {
      std::cout << "Couldn't write to file " << offsets_name << std::endl;
      std::exit(-5);
    }

    std::fstream neighbors_out(neighbors_name,
                               std::ios::out | std::ios::binary);
    if (!neighbors_out) {
      std::cout << "Couldn't write to file " << neighbors_name << std::endl;
      std::exit(-5);
    }

    pvector<SGOffset> offsets = g.VertexOffsets(false);
    offsets_out.write(reinterpret_cast<const char*>(offsets.data()),
                      offsets.size() * sizeof(SGOffset));
    WriteNeighbors(g, neighbors_out);

    offsets_out.close();
    neighbors_out.close();

    // Pad binary files to power-of-two sizes for SSD alignment.
    uint64_t offsets_filesize = pad_file_to_pow2(offsets_name);
    uint64_t neighbors_filesize = pad_file_to_pow2(neighbors_name);

    uint64_t weights_filesize = 0;
    WriteWeightsIfPresent(g, prefix, weights_filesize);
    WriteGraphInfo(g, prefix, offsets_filesize, neighbors_filesize,
                   weights_filesize);
  }

 private:
  template <typename EdgeT = DestID_>
  static typename std::enable_if<std::is_same<EdgeT, NodeID_>::value,
                                 bool>::type
  HasWeights() {
    return false;
  }

  template <typename EdgeT = DestID_>
  static typename std::enable_if<!std::is_same<EdgeT, NodeID_>::value,
                                 bool>::type
  HasWeights() {
    return true;
  }

  static void WriteGraphInfo(const CSRGraph<NodeID_, DestID_> &g,
                             const std::string &prefix,
                             uint64_t offsets_filesize,
                             uint64_t neighbors_filesize,
                             uint64_t weights_filesize) {
    std::string info_name = prefix + ".info.txt";
    std::fstream info_out(info_name, std::ios::out);
    if (!info_out) {
      std::cout << "Couldn't write to file " << info_name << std::endl;
      std::exit(-5);
    }

    info_out << "prefix= " << prefix << std::endl;
    info_out << "directed= " << (g.directed() ? 1 : 0) << std::endl;
    info_out << "weighted= " << (HasWeights() ? 1 : 0) << std::endl;
    info_out << "num_nodes= " << g.num_nodes() << std::endl;
    info_out << "num_edges= " << g.num_edges() << std::endl;
    info_out << "num_edges_directed= " << g.num_edges_directed() << std::endl;
    info_out << "node_id_size= " << sizeof(NodeID_) << std::endl;
    info_out << "offset_size= " << sizeof(SGOffset) << std::endl;
    info_out << "offsets_file= " << prefix << ".offsets.bin" << std::endl;
    info_out << "offsets_filesize= " << offsets_filesize << std::endl;
    info_out << "neighbors_file= " << prefix << ".neighbors.bin" << std::endl;
    info_out << "neighbors_filesize= " << neighbors_filesize << std::endl;
    if (HasWeights()) {
      info_out << "weights_file= " << prefix << ".weights.bin" << std::endl;
      info_out << "weights_filesize= " << weights_filesize << std::endl;
    }

    info_out.close();
  }

  template <typename EdgeT = DestID_>
  static typename std::enable_if<std::is_same<EdgeT, NodeID_>::value,
                                 void>::type
  WriteNeighbors(const CSRGraph<NodeID_, DestID_> &g,
                 std::fstream &neighbors_out) {
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.out_neigh(u))
        neighbors_out.write(reinterpret_cast<const char*>(&v), sizeof(NodeID_));
    }
  }

  template <typename EdgeT = DestID_>
  static typename std::enable_if<!std::is_same<EdgeT, NodeID_>::value,
                                 void>::type
  WriteNeighbors(const CSRGraph<NodeID_, DestID_> &g,
                 std::fstream &neighbors_out) {
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (const EdgeT &edge : g.out_neigh(u)) {
        NodeID_ v = edge.v;
        neighbors_out.write(reinterpret_cast<const char*>(&v), sizeof(NodeID_));
      }
    }
  }

  template <typename EdgeT = DestID_>
  static typename std::enable_if<std::is_same<EdgeT, NodeID_>::value,
                                 void>::type
  WriteWeightsIfPresent(const CSRGraph<NodeID_, DestID_> &,
                        const std::string &, uint64_t &) {}

  template <typename EdgeT = DestID_>
  static typename std::enable_if<!std::is_same<EdgeT, NodeID_>::value,
                                 void>::type
  WriteWeightsIfPresent(const CSRGraph<NodeID_, DestID_> &g,
                        const std::string &prefix,
                        uint64_t &weights_filesize_out) {
    typedef decltype(EdgeT().w) WeightType;

    std::string weights_name = prefix + ".weights.bin";
    std::fstream weights_out(weights_name, std::ios::out | std::ios::binary);
    if (!weights_out) {
      std::cout << "Couldn't write to file " << weights_name << std::endl;
      std::exit(-5);
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (const EdgeT &edge : g.out_neigh(u)) {
        WeightType weight = edge.w;
        weights_out.write(reinterpret_cast<const char*>(&weight),
                          sizeof(WeightType));
      }
    }

    weights_out.close();
    weights_filesize_out = pad_file_to_pow2(weights_name);
  }
};


template <typename NodeID_, typename DestID_>
void DumpGraphCSR(const CSRGraph<NodeID_, DestID_> &g,
                  const std::string &prefix) {
  GraphDumper<NodeID_, DestID_>::DumpCSR(g, prefix);
}


inline bool GraphDumpEnvEnabled(const char *env_name) {
  const char *value = std::getenv(env_name);
  if ((value == nullptr) || (value[0] == '\0'))
    return false;

  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });

  if ((normalized == "1") || (normalized == "true") ||
      (normalized == "yes") || (normalized == "on")) {
    return true;
  }
  if ((normalized == "0") || (normalized == "false") ||
      (normalized == "no") || (normalized == "off")) {
    return false;
  }

  std::cout << "Invalid value for " << env_name << ": " << value << std::endl;
  std::exit(-5);
}


template <typename NodeID_, typename DestID_>
bool MaybeDumpGraphCSRFromEnv(const CSRGraph<NodeID_, DestID_> &g,
                              const char *prefix_env_name =
                                  "GAPBS_CSR_DUMP_PREFIX",
                              const char *enable_env_name =
                                  "GAPBS_CSR_DUMP") {
  if (!GraphDumpEnvEnabled(enable_env_name))
    return false;

  const char *prefix = std::getenv(prefix_env_name);
  if ((prefix == nullptr) || (prefix[0] == '\0')) {
    std::cout << "Environment variable " << prefix_env_name
              << " must be set when " << enable_env_name
              << " enables graph dumping" << std::endl;
    std::exit(-5);
  }

  DumpGraphCSR(g, std::string(prefix));
  return true;
}

#endif  // GRAPH_DUMP_H_