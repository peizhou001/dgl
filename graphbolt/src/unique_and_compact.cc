/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/unique_and_compact.h>

#include <unordered_map>

#include "./concurrent_id_hash_map.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids) {
  torch::Tensor compacted_src_ids;
  torch::Tensor compacted_dst_ids;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  torch::Tensor ids = torch::cat({unique_dst_ids, src_ids});
  AT_DISPATCH_INTEGRAL_TYPES(
      ids.scalar_type(), "unique_and_compact", ([&] {
// TODO: Remove this after windows concurrent bug being fixed.
#ifdef _MSC_VER
        std::unordered_map<scalar_t> id_map;
        auto ids_data = ids.data_ptr<scalar_t>();
        scalar_t index = 0;
        for (auto id : ids) {
          if (id_map.count(id) == 0) {
            id_map[id] = index++;
          }
        }
        compacted_src_ids = torch::empty_like(src_ids);
        compacted_dst_ids = torch::empty_like(dst_ids);
        auto num_ids = compacted_src_ids.size(0);
        auto src_ids_data = src_ids.data_ptr<scalar_t>();
        auto dst_ids_data = dst_ids.data_ptr<scalar_t>();
        auto compacted_src_ids_data = compacted_src_ids.data_ptr<scalar_t>();
        auto compacted_dst_ids_data = compacted_dst_ids.data_ptr<scalar_t>();
        torch::parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
          for (int64_t i = s; i < e; i++) {
            compacted_src_ids_data[i] = id_map(src_ids_data[i]);
            compacted_dst_ids_data[i] = id_map(dst_ids_data[i]);
          }
        });
#else
        ConcurrentIdHashMap<scalar_t> id_map;
        unique_ids = id_map.Init(ids, num_dst);
        compacted_src_ids = id_map.MapIds(src_ids);
        compacted_dst_ids = id_map.MapIds(dst_ids);
#endif
      }));
  return std::tuple(unique_ids, compacted_src_ids, compacted_dst_ids);
}
}  // namespace sampling
}  // namespace graphbolt
