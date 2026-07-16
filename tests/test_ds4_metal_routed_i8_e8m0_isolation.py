#!/usr/bin/env python3
"""Real-payload Metal-vs-OCP witness proof for routed I8+F8_E8M0.

This test is intentionally independent of the incumbent Python dequant path: it
uses the existing safetensors-header reader only to locate real checkpoint byte
slices, then compares the Metal proof kernel against the standalone OCP-spec
witness in tests/ds4_e8m0_ocp_witness.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from array import array
from pathlib import Path

from tests.ds4_e8m0_ocp_witness import decode_i8_e8m0_to_float32

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))

from ds4_ft_mlx.deepseek_v4_dequant import read_safetensors_header  # noqa: E402

DEFAULT_HF_MODEL = Path(
    "/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
    "snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0"
)
FORWARD_PARITY_MARKER = Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok")


RUNNER_SOURCE = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t out_stride;
    uint32_t pad;
} ds4_test_dequant_args;

static NSData *read_file(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
        fprintf(stderr, "failed to read %s\n", [path UTF8String]);
        exit(2);
    }
    return data;
}

static NSString *read_text(NSString *path) {
    NSError *error = nil;
    NSString *text = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
    if (!text) {
        fprintf(stderr, "failed to read %s: %s\n", [path UTF8String], [[error localizedDescription] UTF8String]);
        exit(2);
    }
    return text;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s REPO_ROOT WEIGHT SCALE ROWS COLS OUT_BF16\n", argv[0]);
        return 2;
    }

    @autoreleasepool {
        NSString *repo = [NSString stringWithUTF8String:argv[1]];
        NSString *weight_path = [NSString stringWithUTF8String:argv[2]];
        NSString *scale_path = [NSString stringWithUTF8String:argv[3]];
        uint32_t rows = (uint32_t)strtoul(argv[4], NULL, 10);
        uint32_t cols = (uint32_t)strtoul(argv[5], NULL, 10);
        NSString *out_path = [NSString stringWithUTF8String:argv[6]];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal device unavailable\n");
            return 77;
        }

        NSString *base = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
                          "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
                          "#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
                          "#define QK8_0 32\n"
                          "#define N_SIMDWIDTH 32\n"
                          "#define N_R0_Q8_0 2\n"
                          "#define N_SG_Q8_0 4\n"
                          "#define FC_MUL_MV 600\n"
                          "#define FC_MUL_MM 700\n"
                          "#define FC_BIN 1300\n"
                          "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
                          "#define M_PI_F 3.14159265358979323846f\n"
                          "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
                          "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n";
        NSMutableString *source = [NSMutableString stringWithString:base];
        NSArray<NSString *> *metalFiles = @[
            @"metal/flash_attn.metal",
            @"metal/dense.metal",
            @"metal/moe.metal",
        ];
        for (NSString *metalFile in metalFiles) {
            [source appendString:@"\n"];
            [source appendString:read_text([repo stringByAppendingPathComponent:metalFile])];
        }

        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "Metal compile failed: %s\n", [[error localizedDescription] UTF8String]);
            return 3;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_dsv4_routed_dequant_i8_e8m0_to_bf16"];
        if (!fn) {
            fprintf(stderr, "proof kernel not found\n");
            return 4;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            return 5;
        }

        NSData *weight = read_file(weight_path);
        NSData *scale = read_file(scale_path);
        uint64_t elements = (uint64_t)rows * (uint64_t)cols;
        uint64_t out_bytes = elements * sizeof(uint16_t);
        if ([weight length] != elements || [scale length] != (uint64_t)rows * (uint64_t)(cols / 16u)) {
            fprintf(stderr, "unexpected tensor byte lengths: weight=%llu scale=%llu rows=%u cols=%u\n",
                    (unsigned long long)[weight length], (unsigned long long)[scale length], rows, cols);
            return 6;
        }

        ds4_test_dequant_args args = { rows, cols, cols, 0 };
        id<MTLBuffer> args_buf = [device newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buf = [device newBufferWithBytes:[weight bytes] length:[weight length] options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale_buf = [device newBufferWithBytes:[scale bytes] length:[scale length] options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [device newBufferWithLength:(NSUInteger)out_bytes options:MTLResourceStorageModeShared];
        if (!args_buf || !weight_buf || !scale_buf || !out_buf) {
            fprintf(stderr, "buffer allocation failed\n");
            return 7;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:args_buf offset:0 atIndex:0];
        [enc setBuffer:weight_buf offset:0 atIndex:1];
        [enc setBuffer:scale_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        NSUInteger tg = MIN((NSUInteger)256, [pipeline maxTotalThreadsPerThreadgroup]);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)elements, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "command buffer failed: %s\n", [[[[cmd error] localizedDescription] description] UTF8String]);
            return 8;
        }

        NSData *out = [NSData dataWithBytes:[out_buf contents] length:(NSUInteger)out_bytes];
        if (![out writeToFile:out_path atomically:NO]) {
            fprintf(stderr, "failed to write output %s\n", [out_path UTF8String]);
            return 9;
        }
    }
    return 0;
}
'''


def _hf_model_dir() -> Path:
    return Path(os.environ.get("DS4_HF_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL)


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise unittest.SkipTest(f"real DeepSeek V4 Flash index not present at {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AssertionError(f"{index_path}: malformed or missing weight_map")
    return weight_map


def _first_routed_expert_triplet(weight_map: dict[str, str]) -> list[str]:
    grouped: dict[tuple[int, int], dict[str, str]] = {}
    pattern = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
    for name in weight_map:
        match = pattern.match(name)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        grouped.setdefault(key, {})[match.group(3)] = name
    for key in sorted(grouped):
        tensors = grouped[key]
        if all(f"w{i}" in tensors for i in (1, 2, 3)):
            return [tensors["w1"], tensors["w3"], tensors["w2"]]
    raise AssertionError("real checkpoint index contains no complete routed expert w1/w2/w3 triplet")


def _read_safetensors_slice(shard: Path, offsets: list[int]) -> bytes:
    with shard.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        start, end = offsets
        f.seek(8 + header_len + start)
        data = f.read(end - start)
    if len(data) != end - start:
        raise AssertionError(f"{shard}: short read for data_offsets={offsets}")
    return data


def _load_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> tuple[bytes, dict[str, object]]:
    shard = model_dir / weight_map[name]
    header = read_safetensors_header(shard, max_bytes=128 * 1024 * 1024)
    meta = header[name]
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise AssertionError(f"{name}: invalid data_offsets")
    return _read_safetensors_slice(shard, offsets), meta


def _compile_runner(tmp: Path) -> Path:
    clang = shutil.which("clang")
    if clang is None:
        raise unittest.SkipTest("clang unavailable for Objective-C Metal test runner")
    source = tmp / "ds4_e8m0_runner.m"
    binary = tmp / "ds4_e8m0_runner"
    source.write_text(RUNNER_SOURCE, encoding="utf-8")
    cmd = [
        clang,
        "-fobjc-arc",
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        str(source),
        "-o",
        str(binary),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise AssertionError(f"runner compile failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return binary


def _run_metal_decode(runner: Path, tmp: Path, name: str, weight: bytes, scales: bytes, rows: int, cols: int) -> bytes:
    safe = name.replace(".", "_")
    weight_path = tmp / f"{safe}.weight.i8"
    scale_path = tmp / f"{safe}.scale.e8m0"
    out_path = tmp / f"{safe}.out.bf16"
    weight_path.write_bytes(weight)
    scale_path.write_bytes(scales)
    cmd = [str(runner), str(REPO_ROOT), str(weight_path), str(scale_path), str(rows), str(cols), str(out_path)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 77:
        raise unittest.SkipTest("Metal device unavailable")
    if result.returncode != 0:
        raise AssertionError(f"Metal decode runner failed for {name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return out_path.read_bytes()


def _bf16_bytes_to_float32(raw: bytes) -> array:
    if len(raw) % 2:
        raise AssertionError("BF16 byte buffer length is not even")
    words = array("H")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    bits = array("I", (int(word) << 16 for word in words))
    floats = array("f")
    floats.frombytes(bits.tobytes())
    if sys.byteorder != "little":
        floats.byteswap()
    return floats


def _max_abs(actual: array, expected: array) -> float:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: {len(actual)} != {len(expected)}")
    max_abs = 0.0
    for got, want in zip(actual, expected, strict=True):
        if got != got or want != want:
            if got != got and want != want:
                continue
            return float("inf")
        diff = abs(float(got) - float(want))
        if diff > max_abs:
            max_abs = diff
    return max_abs


class DS4MetalRoutedI8E8M0IsolationTests(unittest.TestCase):
    def test_real_hf_routed_i8_e8m0_metal_matches_ocp_witness(self) -> None:
        model_dir = _hf_model_dir()
        weight_map = _load_weight_map(model_dir)
        triplet = _first_routed_expert_triplet(weight_map)

        with tempfile.TemporaryDirectory(prefix="ds4-e8m0-metal-") as td:
            tmp = Path(td)
            runner = _compile_runner(tmp)
            worst = 0.0
            checked = []
            for weight_name in triplet:
                scale_name = weight_name[:-len(".weight")] + ".scale"
                self.assertIn(scale_name, weight_map, f"missing scale tensor for {weight_name}")

                weight, weight_meta = _load_tensor(model_dir, weight_map, weight_name)
                scales, scale_meta = _load_tensor(model_dir, weight_map, scale_name)
                self.assertEqual(weight_meta["dtype"], "I8")
                self.assertEqual(scale_meta["dtype"], "F8_E8M0")
                rows, cols = [int(x) for x in weight_meta["shape"]]
                scale_rows, scale_cols = [int(x) for x in scale_meta["shape"]]
                self.assertEqual(scale_rows, rows)
                self.assertEqual(scale_cols, cols // 16)
                self.assertEqual(cols % 16, 0)
                if weight_name.endswith(".w1.weight"):
                    self.assertEqual([rows, cols], [2048, 2048])
                    self.assertEqual([scale_rows, scale_cols], [2048, 128])

                metal_bf16 = _run_metal_decode(runner, tmp, weight_name, weight, scales, rows, cols)
                metal = _bf16_bytes_to_float32(metal_bf16)
                witness = decode_i8_e8m0_to_float32(weight, scales, rows=rows, cols=cols, block_size=16)
                err = _max_abs(metal, witness)
                worst = max(worst, err)
                checked.append((weight_name, rows, cols, err))

            self.assertLessEqual(worst, 1.0e-5, f"checked={checked}")

        self.assertFalse(FORWARD_PARITY_MARKER.exists(), f"{FORWARD_PARITY_MARKER} must remain absent")


if __name__ == "__main__":
    unittest.main()
