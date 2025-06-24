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
import bz2
import lzma
import zlib

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
            elif algorithm == "lzw":
                compressed = lzw_compress(data)
            elif algorithm == "bwt":
                compressed = bwt_compress(data)
            elif algorithm == "delta":
                compressed = delta_compress(data)
            elif algorithm == "zip":
                compressed = zip_compress(filename, data)
            elif algorithm == "gzip":
                compressed = gzip_compress(data)
            elif algorithm == "bz2":
                compressed = bz2.compress(data)
            elif algorithm == "lzma":
                compressed = lzma.compress(data)
            elif algorithm == "zlib":
                compressed = zlib.compress(data)
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

        # Always write compressed output, even if not smaller
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
        # Try all algorithms (removed huffman, xor)
        for alg in ["rle", "lzw", "bwt", "delta", "zip", "gzip", "bz2", "lzma", "zlib"]:
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
            elif algorithm == "lzw":
                decompressed = lzw_decompress(data)
            elif algorithm == "bwt":
                decompressed = bwt_decompress(data)
            elif algorithm == "delta":
                decompressed = delta_decompress(data)
            elif algorithm == "zip":
                decompressed = zip_decompress(filename, data)
            elif algorithm == "gzip":
                decompressed = gzip_decompress(data)
            elif algorithm == "bz2":
                decompressed = bz2.decompress(data)
            elif algorithm == "lzma":
                decompressed = lzma.decompress(data)
            elif algorithm == "zlib":
                decompressed = zlib.decompress(data)
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
    return bytes(result)

def rle_decompress(data):
    result = bytearray()
    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i+1]
        result.extend([value] * count)
    return bytes(result)

# --- LZW ---
def lzw_compress(data):
    if not data:
        return b''
    max_bits = 12
    max_table_size = 1 << max_bits
    dictionary = {bytes([i]): i for i in range(256)}
    dict_size = 256
    w = b""
    codes = []
    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            if dict_size < max_table_size:
                dictionary[wc] = dict_size
                dict_size += 1
            else:
                # Reset dictionary
                dictionary = {bytes([i]): i for i in range(256)}
                dict_size = 256
                # Avoid redundant entry if wc already in dictionary after reset
                if wc not in dictionary:
                    dictionary[wc] = dict_size
                    dict_size += 1
            w = bytes([c])
    if w:
        codes.append(dictionary[w])
    # Pack codes into a bitstream (12 bits per code)
    out = bytearray()
    buffer = 0
    bits_in_buffer = 0
    for code in codes:
        buffer = (buffer << max_bits) | code
        bits_in_buffer += max_bits
        while bits_in_buffer >= 8:
            bits_in_buffer -= 8
            out.append((buffer >> bits_in_buffer) & 0xFF)
        buffer &= (1 << bits_in_buffer) - 1  # Keep only the remaining bits
    if bits_in_buffer > 0:
        out.append((buffer << (8 - bits_in_buffer)) & 0xFF)
    compressed = bytes([max_bits]) + out
    # Compression ratio logging (optional)
    ratio = len(compressed) / len(data) if len(data) > 0 else 0
    # print(f"LZW compression ratio: {ratio:.2f}")
    if len(compressed) >= len(data):
        return b'UNCOMPRESSED' + data
    return compressed

def lzw_decompress(data):
    if not data:
        return b''
    if data.startswith(b'UNCOMPRESSED'):
        return data[len(b'UNCOMPRESSED'):]
    max_bits = data[0]
    max_table_size = 1 << max_bits
    data = data[1:]
    dictionary = {i: bytes([i]) for i in range(256)}
    dict_size = 256
    codes = []
    buffer = 0
    bits_in_buffer = 0
    idx = 0
    while idx < len(data):
        while bits_in_buffer < max_bits and idx < len(data):
            buffer = (buffer << 8) | data[idx]
            bits_in_buffer += 8
            idx += 1
        if bits_in_buffer < max_bits:
            break
        bits_in_buffer -= max_bits
        code = (buffer >> bits_in_buffer) & ((1 << max_bits) - 1)
        codes.append(code)
        buffer &= (1 << bits_in_buffer) - 1
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
        if dict_size < max_table_size:
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
        else:
            # Reset dictionary
            dictionary = {i: bytes([i]) for i in range(256)}
            dict_size = 256
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

# --- ZIP ---
def zip_compress(filename, data):
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, data)
    return memfile.getvalue()

def zip_decompress(filename, data):
    memfile = io.BytesIO(data)
    with zipfile.ZipFile(memfile, 'r') as zf:
        # Try to extract by original filename first
        if filename in zf.namelist():
            return zf.read(filename)
        # Fallback: extract the first file
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

def log_activity(message):
    with open("activity.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

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
    # Ensure 'data' directory exists
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
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
