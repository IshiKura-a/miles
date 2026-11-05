"""Build a unified (market, docid, keyword) columnar index from NA and INTL SStream (.ss) tables.

Decodes Cosmos Structured Stream (MTSS Format 1) partitions in parallel, preserves
exact keyword bytes without stripping or case folding, unions allowed markets per
unique keyword into a static `keywords.parquet` table, and writes deduplicated
sorted `(docid_u32, keyword_id)` pairs into a dynamic `docid_keywords.parquet` table.

Example:
    uv run --env-file .env python -m examples.experimental.generative_retriever.index.build_sstream_index \
        --na-ss "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_NATAPart.2026-10-28.ss" \
        --intl-ss "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_INTLTAPart.2026-10-29.ss" \
        --output "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/slmgrv3_part25_na_intl_20261028_20261029"
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_NUM_SHARDS = 64

_CPP_WORKER_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

static inline uint64_t fnv1a_64(const char* data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint32_t read_varint(const uint8_t* buf, size_t& pos) {
    uint32_t val = 0;
    uint32_t shift = 0;
    while (true) {
        uint8_t b = buf[pos++];
        val |= static_cast<uint32_t>(b & 0x7F) << shift;
        if (!(b & 0x80)) return val;
        shift += 7;
    }
}

static inline bool parse_docid_u32(const char* s, size_t len, uint32_t& out) {
    int vals[4] = {0, 0, 0, 0};
    int idx = 0;
    bool in_num = false;
    int cur = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = s[i];
        if (c >= '0' && c <= '9') {
            cur = cur * 10 + (c - '0');
            in_num = true;
        } else if (c == ' ') {
            if (!in_num || idx >= 4) return false;
            vals[idx++] = cur;
            cur = 0;
            in_num = false;
        } else {
            return false;
        }
    }
    if (!in_num || idx != 3) return false;
    vals[3] = cur;
    for (int p = 0; p < 4; ++p) {
        if (vals[p] < p * 256 || vals[p] >= (p + 1) * 256) return false;
    }
    out = (static_cast<uint32_t>(vals[0]) << 24) |
          (static_cast<uint32_t>(vals[1] - 256) << 16) |
          (static_cast<uint32_t>(vals[2] - 512) << 8) |
          static_cast<uint32_t>(vals[3] - 768);
    return true;
}

extern "C" int partition_to_shards(
    const char* ss_path,
    int64_t part_offset,
    int64_t expected_rows,
    int num_cols,
    const char* shard_prefix,
    int num_shards
) {
    FILE* f = std::fopen(ss_path, "rb");
    if (!f) return -1;
    if (std::fseek(f, part_offset, SEEK_SET) != 0) {
        std::fclose(f);
        return -2;
    }

    std::vector<FILE*> out_files(num_shards, nullptr);
    std::vector<std::vector<uint8_t>> out_bufs(num_shards);
    for (int s = 0; s < num_shards; ++s) {
        char path[1024];
        std::snprintf(path, sizeof(path), "%s.shard_%02d.bin", shard_prefix, s);
        out_files[s] = std::fopen(path, "wb");
        if (!out_files[s]) return -3;
        out_bufs[s].reserve(1 << 20);
    }

    auto flush_shard = [&](int s) {
        if (!out_bufs[s].empty()) {
            std::fwrite(out_bufs[s].data(), 1, out_bufs[s].size(), out_files[s]);
            out_bufs[s].clear();
        }
    };

    std::vector<uint8_t> blk;
    std::vector<uint32_t> l0, l1, l2;
    int64_t rows_read = 0;

    while (rows_read < expected_rows) {
        uint32_t hdr[3];
        if (std::fread(hdr, sizeof(uint32_t), 3, f) != 3) return -4;
        uint32_t ver = hdr[0], blk_len = hdr[1], row_cnt = hdr[2];
        if (ver != 0x0103 || blk_len < 12) return -5;
        blk.resize(blk_len);
        std::memcpy(blk.data(), hdr, 12);
        if (std::fread(blk.data() + 12, 1, blk_len - 12, f) != blk_len - 12) return -6;

        const uint32_t* descs = reinterpret_cast<const uint32_t*>(blk.data() + blk_len - 4 * num_cols);
        size_t pos0 = descs[0] & 0x00FFFFFF;
        uint32_t start0 = read_varint(blk.data(), pos0);
        l0.resize(row_cnt);
        for (uint32_t i = 0; i < row_cnt; ++i) l0[i] = read_varint(blk.data(), pos0);

        size_t pos1 = descs[1] & 0x00FFFFFF;
        uint32_t start1 = read_varint(blk.data(), pos1);
        l1.resize(row_cnt);
        for (uint32_t i = 0; i < row_cnt; ++i) l1[i] = read_varint(blk.data(), pos1);

        uint32_t start2 = 0;
        if (num_cols == 3) {
            size_t pos2 = descs[2] & 0x00FFFFFF;
            start2 = read_varint(blk.data(), pos2);
            l2.resize(row_cnt);
            for (uint32_t i = 0; i < row_cnt; ++i) l2[i] = read_varint(blk.data(), pos2);
        }

        const char* p0 = reinterpret_cast<const char*>(blk.data() + start0);
        const char* p1 = reinterpret_cast<const char*>(blk.data() + start1);
        const char* p2 = num_cols == 3 ? reinterpret_cast<const char*>(blk.data() + start2) : nullptr;

        for (uint32_t i = 0; i < row_cnt; ++i) {
            const char* did_ptr = num_cols == 2 ? p0 : p1;
            uint32_t did_len = num_cols == 2 ? l0[i] : l1[i];
            const char* kw_ptr = num_cols == 2 ? p1 : p2;
            uint32_t kw_len = num_cols == 2 ? l1[i] : l2[i];
            const char* mkt_ptr = num_cols == 2 ? nullptr : p0;
            uint32_t mkt_len = num_cols == 2 ? 0 : l0[i];

            p0 += l0[i];
            p1 += l1[i];
            if (num_cols == 3) p2 += l2[i];

            uint32_t docid_u32 = 0;
            if (!parse_docid_u32(did_ptr, did_len, docid_u32)) return -7;
            if (kw_len == 0 || kw_len > 65535 || mkt_len > 65535) return -8;

            int shard = static_cast<int>(fnv1a_64(kw_ptr, kw_len) % static_cast<uint64_t>(num_shards));
            auto& buf = out_bufs[shard];
            if (buf.size() + 8 + kw_len + mkt_len > (1 << 20)) {
                flush_shard(shard);
            }
            uint16_t kw_u16 = static_cast<uint16_t>(kw_len);
            uint16_t mkt_u16 = static_cast<uint16_t>(mkt_len);
            size_t old_sz = buf.size();
            buf.resize(old_sz + 8 + kw_len + mkt_len);
            std::memcpy(buf.data() + old_sz, &docid_u32, 4);
            std::memcpy(buf.data() + old_sz + 4, &kw_u16, 2);
            std::memcpy(buf.data() + old_sz + 6, &mkt_u16, 2);
            std::memcpy(buf.data() + old_sz + 8, kw_ptr, kw_len);
            if (mkt_len > 0) {
                std::memcpy(buf.data() + old_sz + 8 + kw_len, mkt_ptr, mkt_len);
            }
        }
        rows_read += row_cnt;
    }

    for (int s = 0; s < num_shards; ++s) {
        flush_shard(s);
        std::fclose(out_files[s]);
    }
    std::fclose(f);
    return 0;
}

struct KwEntry {
    uint32_t local_id;
    bool in_na;
    std::vector<uint16_t> intl_mkt_ids;
};

extern "C" int reduce_shard(
    const char* const* shard_files,
    int num_files,
    const char* out_prefix,
    uint64_t* out_stats
) {
    // Market dictionary (small, < 1000 distinct market codes like EN-GB, JA-JP)
    std::unordered_map<std::string, uint16_t> mkt_to_id;
    std::vector<std::string> id_to_mkt;
    mkt_to_id.reserve(1024);

    auto get_mkt_id = [&](std::string_view sv) -> uint16_t {
        std::string s(sv);
        auto it = mkt_to_id.find(s);
        if (it != mkt_to_id.end()) return it->second;
        uint16_t id = static_cast<uint16_t>(id_to_mkt.size());
        id_to_mkt.push_back(s);
        mkt_to_id.emplace( std::move(s), id );
        return id;
    };

    std::unordered_map<std::string, KwEntry> kw_map;
    kw_map.reserve(5000000);
    std::vector<const std::string*> ordered_kws;
    ordered_kws.reserve(5000000);
    std::vector<uint64_t> pairs; // (uint64_t(docid_u32) << 32) | local_kw_id
    pairs.reserve(8000000);

    uint64_t total_records = 0;
    std::vector<uint8_t> buf(1 << 22);

    for (int f_idx = 0; f_idx < num_files; ++f_idx) {
        FILE* f = std::fopen(shard_files[f_idx], "rb");
        if (!f) return -1;
        while (true) {
            uint8_t hdr[8];
            size_t r = std::fread(hdr, 1, 8, f);
            if (r == 0) break;
            if (r != 8) { std::fclose(f); return -2; }
            uint32_t docid_u32;
            uint16_t kw_len, mkt_len;
            std::memcpy(&docid_u32, hdr, 4);
            std::memcpy(&kw_len, hdr + 4, 2);
            std::memcpy(&mkt_len, hdr + 6, 2);
            size_t payload = static_cast<size_t>(kw_len) + static_cast<size_t>(mkt_len);
            if (buf.size() < payload) buf.resize(payload);
            if (std::fread(buf.data(), 1, payload, f) != payload) { std::fclose(f); return -3; }

            std::string kw_str(reinterpret_cast<const char*>(buf.data()), kw_len);
            auto [it, inserted] = kw_map.try_emplace(std::move(kw_str));
            if (inserted) {
                uint32_t lid = static_cast<uint32_t>(ordered_kws.size());
                it->second.local_id = lid;
                it->second.in_na = false;
                ordered_kws.push_back(&it->first);
            }
            KwEntry& entry = it->second;
            if (mkt_len == 0) {
                entry.in_na = true;
            } else {
                std::string_view mkt_sv(reinterpret_cast<const char*>(buf.data() + kw_len), mkt_len);
                size_t start = 0;
                while (start < mkt_sv.size()) {
                    size_t comma = mkt_sv.find(',', start);
                    if (comma == std::string_view::npos) comma = mkt_sv.size();
                    if (comma > start) {
                        uint16_t mid = get_mkt_id(mkt_sv.substr(start, comma - start));
                        auto& vec = entry.intl_mkt_ids;
                        if (std::find(vec.begin(), vec.end(), mid) == vec.end()) {
                            vec.push_back(mid);
                        }
                    }
                    start = comma + 1;
                }
            }
            pairs.push_back((static_cast<uint64_t>(docid_u32) << 32) | entry.local_id);
            total_records++;
        }
        std::fclose(f);
    }

    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    // Write reduced keyword table and pair array
    std::string kw_data_path = std::string(out_prefix) + ".kw.bin";
    std::string kw_off_path = std::string(out_prefix) + ".kw_off.i64";
    std::string in_na_path = std::string(out_prefix) + ".in_na.u8";
    std::string mkt_data_path = std::string(out_prefix) + ".mkt.bin";
    std::string mkt_off_path = std::string(out_prefix) + ".mkt_off.i64";
    std::string pairs_path = std::string(out_prefix) + ".pairs.u64";

    FILE* f_kw = std::fopen(kw_data_path.c_str(), "wb");
    FILE* f_kw_off = std::fopen(kw_off_path.c_str(), "wb");
    FILE* f_na = std::fopen(in_na_path.c_str(), "wb");
    FILE* f_mkt = std::fopen(mkt_data_path.c_str(), "wb");
    FILE* f_mkt_off = std::fopen(mkt_off_path.c_str(), "wb");
    FILE* f_pairs = std::fopen(pairs_path.c_str(), "wb");
    if (!f_kw || !f_kw_off || !f_na || !f_mkt || !f_mkt_off || !f_pairs) return -4;

    int64_t kw_offset = 0;
    int64_t mkt_offset = 0;
    std::fwrite(&kw_offset, sizeof(int64_t), 1, f_kw_off);
    std::fwrite(&mkt_offset, sizeof(int64_t), 1, f_mkt_off);

    uint64_t na_only = 0, intl_only = 0, both_na_intl = 0;
    std::vector<std::string_view> mkt_names;
    std::string mkt_joined;

    for (const std::string* kw_ptr : ordered_kws) {
        const KwEntry& entry = kw_map.find(*kw_ptr)->second;
        std::fwrite(kw_ptr->data(), 1, kw_ptr->size(), f_kw);
        kw_offset += static_cast<int64_t>(kw_ptr->size());
        std::fwrite(&kw_offset, sizeof(int64_t), 1, f_kw_off);

        uint8_t na_flag = entry.in_na ? 1 : 0;
        std::fwrite(&na_flag, 1, 1, f_na);

        if (entry.in_na && !entry.intl_mkt_ids.empty()) both_na_intl++;
        else if (entry.in_na) na_only++;
        else intl_only++;

        mkt_names.clear();
        for (uint16_t mid : entry.intl_mkt_ids) {
            mkt_names.push_back(id_to_mkt[mid]);
        }
        std::sort(mkt_names.begin(), mkt_names.end());
        mkt_joined.clear();
        if (entry.in_na) {
            mkt_joined.append("NA");
        }
        for (auto sv : mkt_names) {
            if (!mkt_joined.empty()) mkt_joined.push_back(',');
            mkt_joined.append(sv);
        }
        std::fwrite(mkt_joined.data(), 1, mkt_joined.size(), f_mkt);
        mkt_offset += static_cast<int64_t>(mkt_joined.size());
        std::fwrite(&mkt_offset, sizeof(int64_t), 1, f_mkt_off);
    }

    std::fwrite(pairs.data(), sizeof(uint64_t), pairs.size(), f_pairs);

    std::fclose(f_kw);
    std::fclose(f_kw_off);
    std::fclose(f_na);
    std::fclose(f_mkt);
    std::fclose(f_mkt_off);
    std::fclose(f_pairs);

    out_stats[0] = total_records;
    out_stats[1] = ordered_kws.size();
    out_stats[2] = pairs.size();
    out_stats[3] = na_only;
    out_stats[4] = intl_only;
    out_stats[5] = both_na_intl;
    return 0;
}
"""


@dataclass(frozen=True)
class SStreamPartition:
    source_path: Path
    partition_index: int
    offset: int
    data_length: int
    row_count: int
    num_cols: int
    tag: str


def _parse_sstream_partitions(path: Path, *, num_cols: int, tag: str) -> list[SStreamPartition]:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 128 * 1024))
        tail = stream.read()
    idx = tail.find(b"<Name>PartitionIndex</Name>")
    if idx < 0:
        raise ValueError(f"SStream partition schema not found in {path}")
    xml_start = tail.rfind(b"<?xml", 0, idx)
    for pos in range(max(0, xml_start - 4096), xml_start - 12):
        ver, blk_len, row_cnt = struct.unpack_from("<III", tail, pos)
        if ver == 0x0103 and 1 <= row_cnt <= 256 and 12 + 40 * row_cnt <= blk_len < 16384:
            p_idx = struct.unpack_from(f"<{row_cnt}i", tail, pos + 12)
            offsets = struct.unpack_from(f"<{row_cnt}q", tail, pos + 12 + 8 * row_cnt)
            data_lengths = struct.unpack_from(f"<{row_cnt}q", tail, pos + 12 + 24 * row_cnt)
            row_counts = struct.unpack_from(f"<{row_cnt}q", tail, pos + 12 + 32 * row_cnt)
            if list(p_idx) == list(range(row_cnt)) and offsets[0] == 0:
                return [
                    SStreamPartition(
                        source_path=path,
                        partition_index=int(p_idx[i]),
                        offset=int(offsets[i]),
                        data_length=int(data_lengths[i]),
                        row_count=int(row_counts[i]),
                        num_cols=num_cols,
                        tag=tag,
                    )
                    for i in range(row_cnt)
                ]
    raise RuntimeError(f"Failed to locate SStream partition table in {path}")


def _compile_cpp_helper(work_dir: Path) -> Path:
    cpp_path = work_dir / "sstream_worker.cpp"
    so_path = work_dir / "libsstream_worker.so"
    cpp_path.write_text(_CPP_WORKER_SOURCE, encoding="utf-8")
    subprocess.run(
        ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(cpp_path), "-o", str(so_path)],
        check=True,
    )
    return so_path


def _run_partition_worker(args: tuple[str, SStreamPartition, str, int]) -> tuple[str, int, int]:
    so_path, part, shard_dir, num_shards = args
    lib = ctypes.CDLL(so_path)
    lib.partition_to_shards.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.partition_to_shards.restype = ctypes.c_int
    prefix = f"{shard_dir}/{part.tag}_part_{part.partition_index:02d}"
    rc = lib.partition_to_shards(
        os.fsencode(part.source_path),
        part.offset,
        part.row_count,
        part.num_cols,
         prefix.encode("utf-8"),
        num_shards,
    )
    if rc != 0:
        raise RuntimeError(f"partition_to_shards failed for {part.tag} part {part.partition_index}: rc={rc}")
    return part.tag, part.partition_index, part.row_count


def _run_reduce_worker(args: tuple[str, int, list[str], str]) -> dict[str, int]:
    so_path, shard_id, shard_files, reduced_dir = args
    lib = ctypes.CDLL(so_path)
    lib.reduce_shard.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.reduce_shard.restype = ctypes.c_int
    encoded_files = [os.fsencode(p) for p in shard_files]
    c_files = (ctypes.c_char_p * len(encoded_files))(*encoded_files)
    out_prefix = f"{reduced_dir}/reduced_{shard_id:02d}"
    stats = (ctypes.c_uint64 * 6)()
    rc = lib.reduce_shard(c_files, len(encoded_files), out_prefix.encode("utf-8"), stats)
    if rc != 0:
        raise RuntimeError(f"reduce_shard failed for shard {shard_id}: rc={rc}")
    for path in shard_files:
        Path(path).unlink(missing_ok=True)
    return {
        "shard_id": shard_id,
        "total_records": int(stats[0]),
        "unique_keywords": int(stats[1]),
        "unique_pairs": int(stats[2]),
        "na_only": int(stats[3]),
        "intl_only": int(stats[4]),
        "both_na_intl": int(stats[5]),
    }


def build_index(
    *,
    na_ss: Path,
    intl_ss: Path,
    output_dir: Path,
    work_Root: Path = Path("/tmp"),
    workers: int = 32,
) -> dict[str, Any]:
    started = time.monotonic()
    na_ss = na_ss.expanduser().resolve()
    intl_ss = intl_ss.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not na_ss.is_file():
        raise FileNotFoundError(f"NA SStream file not found: {na_ss}")
    if not intl_ss.is_file():
        raise FileNotFoundError(f"INTL SStream file not found: {intl_ss}")

    na_parts = _parse_sstream_partitions(na_ss, num_cols=2, tag="na")
    intl_parts = _parse_sstream_partitions(intl_ss, num_cols=3, tag="intl")
    all_parts = na_parts + intl_parts
    print(
        f"Parsed partitions: NA={len(na_parts)} ({sum(p.row_count for p in na_parts):,} rows), "
        f"INTL={len(intl_parts)} ({sum(p.row_count for p in intl_parts):,} rows)",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="gr_sstream_idx_", dir=work_Root) as tmp_str:
        tmp_dir = Path(tmp_str)
        shard_dir = tmp_dir / "shards"
        reduced_dir = tmp_dir / "reduced"
        local_out = tmp_dir / "out"
        shard_dir.mkdir()
        reduced_dir.mkdir()
        local_out.mkdir()

        so_path = _compile_cpp_helper(tmp_dir)
        t_phase = time.monotonic()
        part_tasks = [(str(so_path), part, str(shard_dir), _NUM_SHARDS) for part in all_parts]
        with ProcessPoolExecutor(max_workers=min(workers, len(part_tasks))) as executor:
            list(executor.map(_run_partition_worker, part_tasks))
        print(f"[Phase 1/3] Partitioned {len(all_parts)} SStream partitions into {_NUM_SHARDS} shards in {time.monotonic() - t_phase:.2f}s", flush=True)

        t_phase = time.monotonic()
        reduce_tasks = []
        for s in range(_NUM_SHARDS):
            files = sorted(str(p) for p in shard_dir.glob(f"*.shard_{s:02d}.bin"))
            reduce_tasks.append((str(so_path), s, files, str(reduced_dir)))
        with ProcessPoolExecutor(max_workers=min(workers, _NUM_SHARDS)) as executor:
            shard_stats = list(executor.map(_run_reduce_worker, reduce_tasks))
        print(f"[Phase 2/3] Reduced {_NUM_SHARDS} shards in {time.monotonic() - t_phase:.2f}s", flush=True)

        t_phase = time.monotonic()
        # Assemble keywords.parquet and global (docid_u32, keyword_id) pairs
        kw_Schema = pa.schema([
            ("keyword_id", pa.uint32()),
            ("keyword", pa.large_binary()),
            ("in_na", pa.bool_()),
            ("markets", pa.large_string()),
        ])
        kw_parquet_path = local_out / "keywords.parquet"
        global_kw_offset = 0
        total_pairs = sum(st["unique_pairs"] for st in shard_stats)
        all_pairs_u64 = np.empty(total_pairs, dtype=np.uint64)
        pair_cursor = 0

        with pq.ParquetWriter(kw_parquet_path, kw_Schema, compression="zstd") as writer:
            for s in range(_NUM_SHARDS):
                prefix = reduced_dir / f"reduced_{s:02d}"
                n_kw = shard_stats[s]["unique_keywords"]
                kw_bytes = Path(f"{prefix}.kw.bin").read_bytes()
                kw_offs = np.fromfile(f"{prefix}.kw_off.i64", dtype=np.int64)
                in_na_arr = np.fromfile(f"{prefix}.in_na.u8", dtype=np.uint8).astype(np.bool_)
                mkt_bytes = Path(f"{prefix}.mkt.bin").read_bytes()
                mkt_offs = np.fromfile(f"{prefix}.mkt_off.i64", dtype=np.int64)

                kw_ids = np.arange(global_kw_offset, global_kw_offset + n_kw, dtype=np.uint32)
                pa_kw = pa.LargeBinaryArray.from_buffers(
                    pa.large_binary(),
                    n_kw,
                    [None, pa.py_buffer(kw_offs), pa.py_buffer(kw_bytes)],
                )
                pa_mkt = pa.LargeStringArray.from_buffers(
                    n_kw,
                    pa.py_buffer(mkt_offs),
                    pa.py_buffer(mkt_bytes),
                )
                batch = pa.Table.from_arrays(
                    [pa.array(kw_ids, type=pa.uint32()), pa_kw, pa.array(in_na_arr), pa_mkt],
                    schema=kw_Schema,
                )
                writer.write_table(batch)

                shard_pairs = np.fromfile(f"{prefix}.pairs.u64", dtype=np.uint64)
                shard_pairs += np.uint64(global_kw_offset)
                all_pairs_u64[pair_cursor : pair_cursor + len(shard_pairs)] = shard_pairs
                pair_cursor += len(shard_pairs)
                global_kw_offset += n_kw

                for ext in (".kw.bin", ".kw_off.i64", ".in_na.u8", ".mkt.bin", ".mkt_off.i64", ".pairs.u64"):
                    Path(f"{prefix}{ext}").unlink(missing_ok=True)

        # Global sort of (docid_u32 << 32) | keyword_id
        all_pairs_u64.sort()
        docid_u32 = (all_pairs_u64 >> np.uint64(32)).astype(np.uint32)
        keyword_id = (all_pairs_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        del all_pairs_u64

        # Verify all 256 roots are covered
        roots = np.unique(docid_u32 >> np.uint32(24))
        if len(roots) != 256:
            raise ValueError(f"Index must cover all 256 root RQ codes, found {len(roots)}")

        # Compute unique docids and CSR offsets
        change_idx = np.flatnonzero(docid_u32[1:] != docid_u32[:-1]) + 1
        unique_docid_u32 = np.empty(len(change_idx) + 1, dtype=np.uint32)
        unique_docid_u32[0] = docid_u32[0]
        unique_docid_u32[1:] = docid_u32[change_idx]
        csr_offsets = np.empty(len(unique_docid_u32) + 1, dtype=np.int64)
        csr_offsets[0] = 0
        csr_offsets[1:-1] = change_idx
        csr_offsets[-1] = len(docid_u32)

        pairs_schema = pa.schema([
            ("docid_u32", pa.uint32()),
            ("keyword_id", pa.uint32()),
        ])
        pairs_parquet_path = local_out / "docid_keywords.parquet"
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array(docid_u32, type=pa.uint32()), pa.array(keyword_id, type=pa.uint32())],
                schema=pairs_schema,
            ),
            pairs_parquet_path,
            compression="zstd",
        )

        # Also save pre-indexed CSR arrays for instant zero-copy mmap startup
        np.save(local_out / "docid_unique.u32.npy", unique_docid_u32)
        np.save(local_out / "docid_offsets.i64.npy", csr_offsets)
        np.save(local_out / "docid_kw_ids.u32.npy", keyword_id)

        summary: dict[str, Any] = {
            "format_version": 1,
            "na_source": str(na_ss),
            "intl_source": str(intl_ss),
            "na_rows": int(sum(p.row_count for p in na_parts)),
            "intl_rows": int(sum(p.row_count for p in intl_parts)),
            "total_input_rows": int(sum(st["total_records"] for st in shard_stats)),
            "unique_keywords": int(global_kw_offset),
            "unique_docids": int(len(unique_docid_u32)),
            "unique_docid_keyword_pairs": int(len(docid_u32)),
            "keywords_na_only": int(sum(st["na_only"] for st in shard_stats)),
            "keywords_intl_only": int(sum(st["intl_only"] for st in shard_stats)),
            "keywords_both_na_and_intl": int(sum(st["both_na_intl"] for st in shard_stats)),
            "keywords_parquet_bytes": kw_parquet_path.stat().st_size,
            "docid_keywords_parquet_bytes": pairs_parquet_path.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        (local_out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[Phase 3/3] Built Parquet + CSR tables in {time.monotonic() - t_phase:.2f}s", flush=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(local_out.iterdir()):
            target = output_dir / item.name
            shutil.copy2(item, target)
            assert target.stat().st_size == item.stat().st_size
        summary["output_dir"] = str(output_dir)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--na-ss", type=Path, required=True)
    parser.add_argument("--intl-ss", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    summary = build_index(
        na_ss=args.na_ss,
        intl_ss=args.intl_ss,
        output_dir=args.output,
        work_Root=args.work_root,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
