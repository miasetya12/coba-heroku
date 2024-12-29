from flask import Flask, jsonify, request

# Membuat aplikasi Flask
app = Flask(__name__)

# Route utama (GET)
@app.route('/')
def home():
    return "Hello, World!"

# API endpoint untuk menerima data JSON (POST)
@app.route('/api', methods=['POST'])
def api():
    if request.is_json:  # Memastikan request berformat JSON
        data = request.get_json()  # Menerima data JSON
        response = {
            'message': 'Data received',
            'received_data': data
        }
        return jsonify(response), 200  # Mengembalikan respons dalam format JSON dan status OK (200)
    else:
        return jsonify({"error": "Request must be JSON"}), 400  # Jika bukan JSON, kembalikan error 400

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
