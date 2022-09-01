// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <string>
#include <vector>
#include <algorithm>

#include "SearchBruteForce.h"
#include "knowhere/archive/BruteForce.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace milvus::query {

SubSearchResult
BruteForceSearch(const dataset::SearchDataset& dataset,
                 const void* chunk_data_raw,
                 int64_t chunk_rows,
                 const BitsetView& bitset,
                 bool traced) {
    SubSearchResult sub_result(dataset.num_queries, dataset.topk, dataset.metric_type, dataset.round_decimal);
    try {
        knowhere::DatasetPtr baseset  = std::make_shared<knowhere::Dataset>();
        knowhere::DatasetPtr queryset = std::make_shared<knowhere::Dataset>();

        baseset->Set(knowhere::meta::ROWS, chunk_rows);
        baseset->Set(knowhere::meta::DIM, dataset.dim);
        baseset->Set(knowhere::meta::TENSOR, chunk_data_raw);

        queryset->Set(knowhere::meta::TENSOR, dataset.query_data);
        queryset->Set(knowhere::meta::ROWS, dataset.num_queries);

        knowhere::Config config;
        knowhere::SetMetaMetricType(config, dataset.metric_type);
        knowhere::SetMetaTopk(config, dataset.topk);
        knowhere::SetMetaTraceVisit(config, true);

        // dataset.
        knowhere::DatasetPtr result = knowhere::BruteForce::Search(baseset, queryset, config, bitset);
        std::copy_n(GetDatasetDistance(result), dataset.num_queries * dataset.topk, sub_result.get_distances());
        std::copy_n( GetDatasetIDs(result), dataset.num_queries * dataset.topk, sub_result.get_seg_offsets());

    } catch (std::exception& e) {
        PanicInfo(e.what());
    }
    sub_result.round_values();
    return sub_result;
}

}  // namespace milvus::query
