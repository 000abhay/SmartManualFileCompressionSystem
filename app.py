from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import threading
import time
from werkzeug.utils import secure_filename
from compression_utils import (
    check_disk_usage,
    compress_file,
    decompress_file,
    get_file_metadata,
    get_storage_metrics,
    calculate_performance_metrics,
    log_activity,
    threaded_compress_file,
)

app = Flask(__name__)
UPLOAD_FOLDER = "data"
DECOMPRESSED_FOLDER = "decompressed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DECOMPRESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Monitor the data folder for new files and compress them automatically
def monitor_data_folder():
    compressed_files = set()
    data_folder = "data"

    while True:
        for filename in os.listdir(data_folder):
            if filename.endswith(".cmp") or filename in compressed_files:
                continue  # Skip already compressed files
            # Default to RLE for auto-compress
            result = compress_file(filename, algorithm="rle")
            if "message" in result:
                try:
                    os.remove(os.path.join(data_folder, filename))  # Remove the original uncompressed file
                    log_activity(f"Compressed and removed original file: {filename}")
                except Exception as e:
                    log_activity(f"Failed to remove original file {filename}: {e}")
            compressed_files.add(filename)
        time.sleep(5)  # Check every 5 seconds

# Start the folder monitoring in a separate thread
threading.Thread(target=monitor_data_folder, daemon=True).start()

# Home Route - Display dashboard
@app.route('/')
def index():
    disk_usage = check_disk_usage()
    storage_metrics = get_storage_metrics()
    performance_metrics = calculate_performance_metrics()
    return render_template(
        'index.html',
        disk_usage=disk_usage,
        storage_metrics=storage_metrics,
        performance_metrics=performance_metrics,
    )

# API Route to upload and compress a file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    algorithm = request.form.get('algorithm', 'rle')
    compression_level = int(request.form.get('compression_level', 5))
    use_thread = request.form.get('use_thread', 'false') == 'true'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if use_thread:
            result = threaded_compress_file(filename, algorithm=algorithm, compression_level=compression_level)
        else:
            result = compress_file(filename, algorithm=algorithm, compression_level=compression_level)
        if "message" in result:
            try:
                os.remove(file_path)  # Remove the original uncompressed file
                log_activity(f"Uploaded, compressed, and removed original file: {filename}")
            except Exception as e:
                log_activity(f"Failed to remove original file {filename}: {e}")
        return jsonify(result)

# API Route to compress file manually
@app.route('/compress', methods=['POST'])
def compress():
    filename = request.form['filename']
    algorithm = request.form.get('algorithm', 'rle')
    compression_level = int(request.form.get('compression_level', 5))
    use_thread = request.form.get('use_thread', 'false') == 'true'
    if use_thread:
        result = threaded_compress_file(filename, algorithm=algorithm, compression_level=compression_level)
    else:
        result = compress_file(filename, algorithm=algorithm, compression_level=compression_level)
    log_activity(f"Manually compressed file: {filename} using {algorithm}")
    return jsonify(result)

# API Route to decompress file manually
@app.route('/decompress', methods=['POST'])
def decompress():
    filename = request.form['filename']
    algorithm = request.form.get('algorithm', None)
    result = decompress_file(filename, algorithm=algorithm, target_folder=DECOMPRESSED_FOLDER)
    if "message" in result:
        log_activity(f"Manually decompressed file: {filename} and removed compressed file.")
    else:
        log_activity(f"Failed to decompress file: {filename}")
    return jsonify(result)

# API Route to check file metadata (size, access time)
@app.route('/file_metadata', methods=['GET'])
def file_metadata():
    filename = request.args.get('filename')
    metadata = get_file_metadata(filename)
    return jsonify(metadata)

# API Route to fetch storage metrics
@app.route('/storage_metrics', methods=['GET'])
def storage_metrics():
    metrics = get_storage_metrics()
    return jsonify(metrics)

# API Route to fetch performance metrics
@app.route('/performance_metrics', methods=['GET'])
def performance_metrics():
    metrics = calculate_performance_metrics()
    return jsonify(metrics)

@app.route('/static/activity.log')
def serve_activity_log():
    return send_from_directory('.', 'activity.log')

if __name__ == '__main__':
    app.run(debug=True)