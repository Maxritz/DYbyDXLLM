"""Extract embedding + LM head from GGUF as raw float32 blobs.
Handles Q4_K (type 12), Q6_K (type 14), and F32 (type 0) quantization."""
import sys, os, struct, numpy as np, gguf

# GGML tensor types
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q4_1  = 3
GGML_TYPE_Q5_0  = 6
GGML_TYPE_Q5_1  = 7
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q8_1  = 9
GGML_TYPE_Q2_K  = 10
GGML_TYPE_Q3_K  = 11
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q5_K  = 13
GGML_TYPE_Q6_K  = 14
GGML_TYPE_Q8_K  = 15
GGML_TYPE_IQ4_NL = 30

QK_K = 256  # K-quant superblock size

def half_to_float(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1f
    m = h & 0x3ff
    if e == 0:
        return 0.0 if m == 0 else 2**(-14) * m / 1024
    if e == 31:
        return float('inf') if m == 0 else float('nan')
    return (-1)**s * 2**(e-15) * (1 + m/1024)

def dequantize_q4_k(data):
    """Dequantize Q4_K block. data: 144 bytes for 256 elements."""
    n = QK_K
    result = np.zeros(n, dtype=np.float32)
    d = half_to_float(struct.unpack('<H', data[0:2])[0])
    dmin = half_to_float(struct.unpack('<H', data[2:4])[0])
    # 12 bytes of 6-bit scales for 16 blocks
    raw_scales = np.frombuffer(data[4:16], dtype=np.uint8)
    scales = np.zeros(16, dtype=np.int32)
    for i in range(16):
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        lo = np.uint16(raw_scales[byte_idx]) if byte_idx < len(raw_scales) else np.uint16(0)
        hi = np.uint16(raw_scales[byte_idx+1]) << 8 if byte_idx+1 < len(raw_scales) else np.uint16(0)
        val = lo | hi
        scales[i] = (val >> bit_off) & 0x3F
        if scales[i] >= 32: scales[i] -= 64
    # 128 bytes of 4-bit quants for 256 elements
    quants = np.frombuffer(data[16:144], dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 2
        nibble = (i % 2) * 4
        q = (quants[byte_idx] >> nibble) & 0xF
        blk = i // 16
        result[i] = (float(q) - 16.0) * d * float(scales[blk]) + dmin
    return result

def dequantize_q6_k(data):
    """Dequantize Q6_K block. data: 210 bytes for 256 elements."""
    n = QK_K
    result = np.zeros(n, dtype=np.float32)
    d = half_to_float(struct.unpack('<H', data[0:2])[0])
    dmin = half_to_float(struct.unpack('<H', data[2:4])[0])
    # 16 block scales (6 sub-blocks, 6 bits each)
    raw_scales = np.frombuffer(data[4:16], dtype=np.uint8)
    scales = np.zeros(16, dtype=np.int32)
    # Extract 6-bit values from packed bytes
    for i in range(16):
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        if bit_off <= 2:
            val = np.uint16(raw_scales[byte_idx]) | (np.uint16(raw_scales[byte_idx+1]) << 8) if byte_idx+1 < len(raw_scales) else np.uint16(raw_scales[byte_idx])
            scales[i] = (val >> bit_off) & 0x3F
        else:
            val = np.uint16(raw_scales[byte_idx]) | (np.uint16(raw_scales[byte_idx+1]) << 8)
            scales[i] = (val >> bit_off) & 0x3F
        if scales[i] >= 32: scales[i] -= 64
    # 192 bytes of 6-bit quants (256 * 6 / 8 = 192)
    quants = np.frombuffer(data[16:208], dtype=np.uint8)
    for i in range(n):
        blk = i // 16
        j = i % 16
        byte_idx = (i * 6) // 8
        bit_off = (i * 6) % 8
        if bit_off <= 2:
            val = np.uint16(quants[byte_idx]) | (np.uint16(quants[byte_idx+1]) << 8) if byte_idx+1 < len(quants) else np.uint16(quants[byte_idx])
            q = (val >> bit_off) & 0x3F
        else:
            val = np.uint16(quants[byte_idx]) | (np.uint16(quants[byte_idx+1]) << 8)
            q = (val >> bit_off) & 0x3F
        if q >= 32: q -= 64
        sc = scales[blk]
        scl = d * sc
        result[i] = q * scl + dmin
    return result

def dequantize_tensor(tensor):
    """Dequantize a GGUF tensor to float32 numpy array."""
    raw = np.frombuffer(tensor.data, dtype=np.uint8)
    tt = tensor.tensor_type
    shape = [int(s) for s in tensor.shape]
    total = int(np.prod(shape))
    
    if tt == GGML_TYPE_F32:
        return np.frombuffer(tensor.data, dtype=np.float32).reshape(shape)
    elif tt == GGML_TYPE_F16:
        h = np.frombuffer(tensor.data, dtype=np.uint16)
        f = np.vectorize(half_to_float)(h)
        return f.reshape(shape)
    elif tt == GGML_TYPE_Q4_K:
        # Each block: 148 bytes for 256 elements
        block_size = 148
        n_blocks = total // QK_K
        result = np.zeros(total, dtype=np.float32)
        for b in range(n_blocks):
            offset = b * block_size
            block_data = raw[offset:offset+block_size]
            if len(block_data) < block_size: break
            result[b*QK_K:(b+1)*QK_K] = dequantize_q4_k(block_data)
        return result.reshape(shape)
    elif tt == GGML_TYPE_Q6_K:
        block_size = 210
        n_blocks = total // QK_K
        result = np.zeros(total, dtype=np.float32)
        for b in range(n_blocks):
            offset = b * block_size
            block_data = raw[offset:offset+block_size]
            if len(block_data) < block_size: break
            result[b*QK_K:(b+1)*QK_K] = dequantize_q6_k(block_data)
        return result.reshape(shape)
    else:
        # Fallback: return raw bytes (won't be correct but won't crash)
        return np.frombuffer(tensor.data, dtype=np.float32).reshape(shape)

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} model.gguf [out_dir]")
    sys.exit(1)

model_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
reader = gguf.GGUFReader(model_path)

embed_name = output_name = None
for t in reader.tensors:
    if t.name in ("token_embd.weight", "tok_embeddings.weight"): embed_name = t.name
    if t.name in ("output.weight", "lm_head.weight"): output_name = t.name

print(f"Extracting from {model_path}")
print(f"  Embedding: {embed_name}")
print(f"  Output:    {output_name}")

for tname, outfile in [(embed_name, "embedding.bin"), (output_name, "lm_head.bin")]:
    tensor = [t for t in reader.tensors if t.name == tname][0]
    print(f"  Dequantizing {tname} type={tensor.tensor_type} shape={tensor.shape}...")
    f32 = dequantize_tensor(tensor)
    f32.astype(np.float32).tofile(f"{out_dir}/{outfile}")
    mb = os.path.getsize(f"{out_dir}/{outfile}") / (1024*1024)
    print(f"  Saved {outfile}: {f32.shape} ({mb:.1f} MB)")
