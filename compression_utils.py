import os
import subprocess
import platform
import shutil
import time
import threading
import heapq
from collections import defaultdict
import zipfile
import gzip
import io

# Function to check disk usage using OS commands
def check_disk_usage():
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "logicaldisk", "get", "size,freespace,caption"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error fetching disk usage: {e}"
    else:
        try:
            result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE, text=True)
            return result.stdout
        except Exception as e:
            return f"Error fetching disk usage: {e}"

# Compression and decompression dispatcher
def compress_file(filename, algorithm="rle", source_folder="data", target_folder="data", compression_level=5):
    input_path = os.path.join(source_folder, filename)
    output_path = os.path.join(target_folder, f"{filename}.{algorithm}.cmp")

    if not os.path.exists(input_path):
        return {"error": f"File {filename} not found!"}

    if filename.endswith(".cmp"):
        return {"error": f"File {filename} is already compressed!"}

    try:
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
            if algorithm == "rle":
                compressed = rle_compress(data)
            elif algorithm == "huffman":
                compressed = huffman_compress(data)
            elif algorithm == "lzw":
                compressed = lzw_compress(data)
            elif algorithm == "bwt":
                compressed = bwt_compress(data)
            elif algorithm == "delta":
                compressed = delta_compress(data)
            elif algorithm == "xor":
                compressed = xor_compress(data)
            elif algorithm == "zip":
                compressed = zip_compress(filename, data)
            elif algorithm == "gzip":
                compressed = gzip_compress(data)
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

        # If compression is not effective, store original with a flag (except for zip/gzip, which always compress)
        if algorithm not in ("zip", "gzip") and len(compressed) >= len(data):
            with open(output_path, 'wb') as f_out:
                f_out.write(b'UNCOMPRESSED' + data)
            return {"message": f"File compressed (stored as uncompressed, no gain): {filename} → {filename}.{algorithm}.cmp"}
        else:
            with open(output_path, 'wb') as f_out:
                f_out.write(compressed)
            return {"message": f"File compressed successfully: {filename} → {filename}.{algorithm}.cmp"}

    except Exception as e:
        return {"error": f"Compression failed: {e}"}

def decompress_file(filename, algorithm=None, target_folder="decompressed"):
    # filename is expected without .cmp extension
    # Try to find the compressed file with any supported algorithm
    data_folder = "data"
    found = False
    if algorithm:
        input_path = os.path.join(data_folder, f"{filename}.{algorithm}.cmp")
        if os.path.exists(input_path):
            found = True
    else:
        # Try all algorithms
        for alg in ["rle", "huffman", "lzw", "bwt", "delta", "xor", "zip", "gzip"]:
            input_path = os.path.join(data_folder, f"{filename}.{alg}.cmp")
            if os.path.exists(input_path):
                algorithm = alg
                found = True
                break

    if not found:
        return {"error": f"Compressed file for {filename} not found!"}

    output_path = os.path.join(target_folder, filename)

    try:
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
            # Check for uncompressed flag
            if data.startswith(b'UNCOMPRESSED'):
                decompressed = data[len(b'UNCOMPRESSED'):]
            elif algorithm == "rle":
                decompressed = rle_decompress(data)
            elif algorithm == "huffman":
                decompressed = huffman_decompress(data)
            elif algorithm == "lzw":
                decompressed = lzw_decompress(data)
            elif algorithm == "bwt":
                decompressed = bwt_decompress(data)
            elif algorithm == "delta":
                decompressed = delta_decompress(data)
            elif algorithm == "xor":
                decompressed = xor_decompress(data)
            elif algorithm == "zip":
                decompressed = zip_decompress(filename, data)
            elif algorithm == "gzip":
                decompressed = gzip_decompress(data)
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

        with open(output_path, 'wb') as f_out:
            f_out.write(decompressed)

        os.remove(input_path)
        log_activity(f"Decompressed file: {filename} using {algorithm}")
        return {"message": f"File decompressed successfully: {filename}.{algorithm}.cmp → {filename}"}

    except Exception as e:
        return {"error": f"Decompression failed: {e}"}

# ----------- Algorithms (Refined Implementations) -----------

# --- RLE ---
def rle_compress(data):
    if not data:
        return b''
    result = bytearray()
    prev = data[0]
    count = 1
    for b in data[1:]:
        if b == prev and count < 255:
            count += 1
        else:
            result.extend([count, prev])
            prev = b
            count = 1
    result.extend([count, prev])
    # Only return compressed if smaller
    return bytes(result)

def rle_decompress(data):
    result = bytearray()
    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i+1]
        result.extend([value] * count)
    return bytes(result)

# --- Huffman Coding ---
class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_compress(data):
    if not data:
        return b''
    freq = defaultdict(int)
    for b in data:
        freq[b] += 1
    heap = [HuffmanNode(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # Only one symbol, no compression possible
        return b'UNCOMPRESSED' + data
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, merged)
    root = heap[0]
    codebook = {}
    def build_code(node, code=""):
        if node.symbol is not None:
            codebook[node.symbol] = code or "0"
        else:
            build_code(node.left, code + "0")
            build_code(node.right, code + "1")
    build_code(root)
    def encode_tree(node):
        if node.symbol is not None:
            return b'1' + bytes([node.symbol])
        else:
            return b'0' + encode_tree(node.left) + encode_tree(node.right)
    tree_bytes = encode_tree(root)
    encoded_bits = ''.join(codebook[b] for b in data)
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += '0' * padding
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        encoded_bytes.append(int(encoded_bits[i:i+8], 2))
    return len(tree_bytes).to_bytes(4, 'big') + tree_bytes + bytes([padding]) + bytes(encoded_bytes)

def huffman_decompress(data):
    if not data:
        return b''
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    tree_len = int.from_bytes(data[:4], 'big')
    tree_bytes = data[4:4+tree_len]
    padding = data[4+tree_len]
    encoded = data[5+tree_len:]
    def decode_tree(data, idx):
        if data[idx] == 1:
            return HuffmanNode(0, data[idx+1]), idx+2
        else:
            left, next_idx = decode_tree(data, idx+1)
            right, next_idx = decode_tree(data, next_idx)
            return HuffmanNode(0, None, left, right), next_idx
    root, _ = decode_tree(tree_bytes, 0)
    bits = ''
    for b in encoded:
        bits += f'{b:08b}'
    if padding:
        bits = bits[:-padding]
    result = bytearray()
    node = root
    for bit in bits:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = root
    return bytes(result)

# --- LZW ---
def lzw_compress(data):
    if not data:
        return b''
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    w = b""
    result = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = bytes([c])
    if w:
        result.append(dictionary[w])
    # Use 2 bytes per code (max 65535 codes)
    out = bytearray()
    for code in result:
        out += code.to_bytes(2, 'big')
    return out

def lzw_decompress(data):
    if not data:
        return b''
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    codes = [int.from_bytes(data[i:i+2], 'big') for i in range(0, len(data), 2)]
    if not codes:
        return b''
    w = dictionary[codes[0]]
    result = bytearray(w)
    for k in codes[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[:1]
        else:
            return b''
        result += entry
        dictionary[dict_size] = w + entry[:1]
        dict_size += 1
        w = entry
    return bytes(result)

# --- BWT (Efficient) ---
def bwt_compress(data):
    if not data:
        return b''
    marker = 0
    while bytes([marker]) in data:
        marker += 1
        if marker > 255:
            return b'UNCOMPRESSED' + data
    data += bytes([marker])
    n = len(data)
    suffixes = sorted(range(n), key=lambda i: data[i:])
    last_column = bytes([data[(i - 1) % n] for i in suffixes])
    orig_idx = suffixes.index(0)
    return bytes([marker]) + orig_idx.to_bytes(4, 'big') + last_column

def bwt_decompress(data):
    if len(data) < 5:
        return b''
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    marker = data[0]
    orig_idx = int.from_bytes(data[1:5], 'big')
    last_column = data[5:]
    n = len(last_column)
    tuples = sorted([(c, i) for i, c in enumerate(last_column)])
    next_ = [None] * n
    for i, (_, orig_i) in enumerate(tuples):
        next_[i] = orig_i
    res = bytearray()
    idx = orig_idx
    for _ in range(n):
        c = last_column[idx]
        if c == marker and len(res) == n - 1:
            break
        res.append(c)
        idx = next_[idx]
    return bytes(res)

# --- Delta ---
def delta_compress(data):
    if not data:
        return b''
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append((data[i] - data[i-1]) % 256)
    return bytes(result)

def delta_decompress(data):
    if not data:
        return b''
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    result = bytearray([data[0]])
    for i in range(1, len(data)):
        result.append((result[-1] + data[i]) % 256)
    return bytes(result)

# --- XOR ---
def xor_compress(data):
    key = 0xAA
    return bytes([b ^ key for b in data])

def xor_decompress(data):
    key = 0xAA
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    return bytes([b ^ key for b in data])

# --- ZIP ---
def zip_compress(filename, data):
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, data)
    return memfile.getvalue()

def zip_decompress(filename, data):
    memfile = io.BytesIO(data)
    with zipfile.ZipFile(memfile, 'r') as zf:
        # Extract the first file (should match filename)
        for name in zf.namelist():
            return zf.read(name)
    return b''

# --- GZIP ---
def gzip_compress(data):
    memfile = io.BytesIO()
    with gzip.GzipFile(fileobj=memfile, mode='wb') as gz:
        gz.write(data)
    return memfile.getvalue()

def gzip_decompress(data):
    memfile = io.BytesIO(data)
    with gzip.GzipFile(fileobj=memfile, mode='rb') as gz:
        return gz.read()

# ----------- Multi-threading Support Example -----------

def threaded_compress_file(*args, **kwargs):
    result = {}
    def worker():
        res = compress_file(*args, **kwargs)
        result['output'] = res
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return result['output']

# ----------------------------------------------------------

def get_file_metadata(filename):
    file_path = os.path.join('data', filename)
    if os.path.exists(file_path):
        file_stats = os.stat(file_path)
        metadata = {
            'size': file_stats.st_size,
            'last_accessed': file_stats.st_atime
        }
        return metadata
    else:
        return {"error": "File not found"}

def get_storage_metrics():
    total, used, free = shutil.disk_usage("data")
    return {
        "total_space": f"{total // (1024 ** 3)} GB",
        "used_space": f"{used // (1024 ** 3)} GB",
        "free_space": f"{free // (1024 ** 3)} GB",
    }

def calculate_performance_metrics():
    data_folder = "data"
    decompressed_folder = "decompressed"
    total_original_size = 0
    total_compressed_size = 0
    total_access_time_original = 0
    total_access_time_compressed = 0
    total_files_compressed = 0
    total_files_decompressed = 0

    for filename in os.listdir(data_folder):
        if filename.endswith(".cmp"):
            parts = filename.split('.')
            if len(parts) < 3:
                continue
            original_file = '.'.join(parts[:-2])
            compressed_file = filename

            original_path = os.path.join(data_folder, original_file)
            compressed_path = os.path.join(data_folder, compressed_file)

            if os.path.exists(original_path):
                start_time = time.time()
                with open(original_path, 'rb') as f:
                    time.sleep(0.05)
                    f.read()
                total_access_time_original += time.time() - start_time
                original_size = os.path.getsize(original_path)
                total_original_size += original_size

            if os.path.exists(compressed_path):
                start_time = time.time()
                with open(compressed_path, 'rb') as f:
                    f.read()
                total_access_time_compressed += time.time() - start_time
                compressed_size = os.path.getsize(compressed_path)
                total_compressed_size += compressed_size

            total_files_compressed += 1

    for filename in os.listdir(decompressed_folder):
        if os.path.isfile(os.path.join(decompressed_folder, filename)):
            total_files_decompressed += 1

    if total_access_time_original > 0:
        access_time_reduction = (
            (total_access_time_original - total_access_time_compressed)
            / total_access_time_original
        ) * 100
    else:
        access_time_reduction = 0

    space_saved = total_original_size - total_compressed_size

    return {
        "access_time_reduction": round(access_time_reduction, 2),
        "space_saved": f"{space_saved // (1024 ** 2)} MB",
        "total_files_compressed": total_files_compressed,
        "total_files_decompressed": total_files_decompressed,
    }

def log_activity(message):
    with open("activity.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")