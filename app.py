from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.json_util import dumps
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
import string
import pickle
from flask_cors import CORS # type: ignore
from datetime import datetime
import nltk
from nltk.stem import SnowballStemmer
import string
# import string
# import nltk
from nltk.corpus import stopwords
import pandas as pd

# Pastikan NLTK komponen yang diperlukan telah di-download
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)
# client= MongoClient('mongodb://mongo:fuYTxZYAznDtdjmAXqfPXMIqwfPbEseK@monorail.proxy.rlwy.net:43049/')
client= MongoClient('mongodb+srv://doadmin:QY27SLx30H65p48E@db-mongodb-sgp1-58847-e89b5eef.mongo.ondigitalocean.com/admin?tls=true&authSource=admin')
# client= MongoClient('mongodb://localhost:27017/')

db = client['makeup_product']
collection =  db['desc_product_full']
# ratings_collection = db['review_product_81k_v']
# user_collection = db['user_information_81k_v']
ratings_collection = db['review_product']
user_collection = db['user_information']
recommendations_collection =  db['relevan_product_survei']
order_collection =  db['order_scenario']
recommendations_collection_2 =  db['relevan_product_survei_2']
order_collection_2 =  db['order_scenario_2']
data = pd.DataFrame(list(ratings_collection.find()))
products = pd.DataFrame(list(collection.find()))
user = pd.DataFrame(list(user_collection.find()))

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get("name")
    username = data.get("username")
    skintone = data.get("skintone")
    skintype = data.get("skintype")
    undertone = data.get("undertone")

    # Validasi jika username sudah ada
    if user_collection.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    # Ambil user_id terakhir
    last_user = user_collection.find_one(sort=[("user_id", -1)])  # Mengambil user dengan user_id tertinggi
    new_user_id = last_user["user_id"] + 1 if last_user else 1  # Jika tidak ada user sebelumnya, mulai dari 1

    # Simpan user baru dengan data tambahan di MongoDB
    user_collection.insert_one({
        "name": name,
        "username": username,
        "user_id": new_user_id,
        "skintone": skintone,
        "skintype": skintype,
        "undertone": undertone
    })
    
    return jsonify({"message": "User registered successfully", "user_id": new_user_id}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")

    # Periksa apakah username ada di database
    if username is None:
        return jsonify({"error": "Username is required"}), 400

    user = user_collection.find_one({"username": username})
    if user:
        return jsonify({"message": "Login successful", "user_id": user["user_id"]}), 200
    else:
        return jsonify({"error": "Username not found"}), 404

@app.route('/reviews/<user_id>', methods=['GET'])
def get_user_reviews(user_id):
    reviews = list(ratings_collection.find({"user_id": int(user_id)}, {"_id": 0, "product_id": 1, "stars": 1}))
    if reviews:
        return jsonify({"reviews": reviews}), 200
    else:
        return jsonify({"message": "No reviews found for this user"}), 404
  
@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    try:
        # Ambil data input dari request JSON
        data = request.get_json()

        # Validasi input
        if not all(key in data for key in ('user_id', 'product_id', 'stars')):
            return jsonify({"error": "Missing required fields"}), 400

        user_id = data['user_id']
        product_id = data['product_id']
        stars = data['stars']

        # Jika ID berbentuk angka, tidak perlu konversi ke ObjectId
        # Pastikan ID adalah angka atau string
        try:
            user_id = int(user_id)  # Mengubah ke integer
            product_id = int(product_id)  # Mengubah ke integer
        except ValueError:
            return jsonify({"error": "User ID and Product ID must be numeric"}), 400

        # Cek apakah produk dan user ada di database
        product = collection.find_one({"product_id": product_id})
        user = user_collection.find_one({"user_id": user_id})

        if not product or not user:
            return jsonify({"error": "User or Product not found"}), 404

        # Menyimpan rating ke dalam ratings_collection
        rating_data = {
            "user_id": user_id,
            "product_id": product_id,
            "stars": stars,
        }

        ratings_collection.insert_one(rating_data)

        return jsonify({"message": "Rating submitted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_order', methods=['POST'])
def save_order():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Validasi data
        if not data or "user_id" not in data or "order" not in data:
            return jsonify({"message": "Invalid input. 'user_id' and 'order' are required."}), 400

        user_id = data["user_id"]
        order = data["order"]

        # Validasi format order
        if not isinstance(order, dict) or len(order) != 7:
            return jsonify({"message": "Invalid 'order'. Must be a dictionary with 7 positions."}), 400

        for key in ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]:
            if key not in order or not order[key]:
                return jsonify({"message": f"Missing or invalid value for position '{key}'."}), 400

        # Ambil timestamp dari data (timestamp yang ada di request body)
        timestamp = data.get('timestamp')

        # Pastikan timestamp ada dan valid
        if not timestamp:
            return jsonify({'error': 'Timestamp is required'}), 400

        # Mengonversi timestamp dari milidetik ke detik, lalu ke format yang lebih mudah dibaca
        readable_timestamp = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')  # Dari milidetik ke detik

        # Buat data untuk disimpan
        order_data = {
            "user_id": user_id,
            "order": order,
            "timestamp": readable_timestamp  # Format timestamp yang lebih mudah dibaca
        }

        # Simpan ke MongoDB dan ambil hasilnya
        result = order_collection.insert_one(order_data)
        saved_order = order_collection.find_one({"_id": result.inserted_id})

        # Konversi _id menjadi string untuk JSON serializable
        saved_order["_id"] = str(saved_order["_id"])

        # Kembalikan respons sukses
        return jsonify({"message": "Order saved successfully", "data": saved_order}), 200

    except Exception as e:
        print(f"Error saving order: {e}")
        return jsonify({"message": "Error saving order", "error": str(e)}), 500

@app.route('/save_recommendation', methods=['POST'])
def save_recommendation():
    # Mengambil data dari body request
    data = request.get_json()

    # Validasi data
    if not data or 'user_id' not in data or 'recommendations' not in data:
        return jsonify({'error': 'Invalid data format'}), 400
    
    user_id = data['user_id']
    recommendations = data['recommendations']
    timestamp = data.get('timestamp')  # Ambil timestamp dari request

    # Pastikan timestamp ada
    if not timestamp:
        return jsonify({'error': 'Timestamp is required'}), 400

    # Mengonversi timestamp ke format yang lebih mudah dibaca
    timestamp = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')  # Dari milidetik ke detik

    # Memastikan recommendations adalah list
    if not isinstance(recommendations, list):
        return jsonify({'error': 'Recommendations must be an array'}), 400

    # Validasi setiap rekomendasi
    for scenario in recommendations:
        if 'scenario' not in scenario or 'products' not in scenario:
            return jsonify({'error': 'Each recommendation must include scenario and products'}), 400
        if not isinstance(scenario['products'], list):
            return jsonify({'error': 'Products must be a list'}), 400
        for product in scenario['products']:
            # Validasi tambahan untuk revOrNot dan order
            if 'product_id' not in product or 'revOrNot' not in product or 'order' not in product:
                return jsonify({'error': 'Each product must include product_id, revOrNot, and order'}), 400
            if not isinstance(product['revOrNot'], (bool, type(None))):
                return jsonify({'error': 'revOrNot must be a boolean or null'}), 400
            if not isinstance(product['order'], int) or not (0 <= product['order'] <= 5):
                return jsonify({'error': 'order must be an integer between 0 and 5'}), 400

    # Simpan data ke MongoDB, sertakan timestamp
    try:
        recommendation_data = {
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': timestamp  # Simpan timestamp yang sudah dikonversi
        }
        recommendations_collection.insert_one(recommendation_data)
        return jsonify({'message': 'Recommendation saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_order_2', methods=['POST'])
def save_order_2():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Validasi data
        if not data or "user_id" not in data or "order" not in data:
            return jsonify({"message": "Invalid input. 'user_id' and 'order' are required."}), 400

        user_id = data["user_id"]
        order = data["order"]

        # Validasi format order
        if not isinstance(order, dict) or len(order) != 7:
            return jsonify({"message": "Invalid 'order'. Must be a dictionary with 7 positions."}), 400

        for key in ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]:
            if key not in order or not order[key]:
                return jsonify({"message": f"Missing or invalid value for position '{key}'."}), 400

        # Ambil timestamp dari data (timestamp yang ada di request body)
        timestamp = data.get('timestamp')

        # Pastikan timestamp ada dan valid
        if not timestamp:
            return jsonify({'error': 'Timestamp is required'}), 400

        # Mengonversi timestamp dari milidetik ke detik, lalu ke format yang lebih mudah dibaca
        readable_timestamp = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')  # Dari milidetik ke detik

        # Buat data untuk disimpan
        order_data = {
            "user_id": user_id,
            "order": order,
            "timestamp": readable_timestamp  # Format timestamp yang lebih mudah dibaca
        }

        # Simpan ke MongoDB dan ambil hasilnya
        result = order_collection_2.insert_one(order_data)
        saved_order = order_collection_2.find_one({"_id": result.inserted_id})

        # Konversi _id menjadi string untuk JSON serializable
        saved_order["_id"] = str(saved_order["_id"])

        # Kembalikan respons sukses
        return jsonify({"message": "Order saved successfully", "data": saved_order}), 200

    except Exception as e:
        print(f"Error saving order: {e}")
        return jsonify({"message": "Error saving order", "error": str(e)}), 500

@app.route('/save_recommendation_2', methods=['POST'])
def save_recommendation_2():
    # Mengambil data dari body request
    data = request.get_json()

    # Validasi data
    if not data or 'user_id' not in data or 'recommendations' not in data:
        return jsonify({'error': 'Invalid data format'}), 400
    
    user_id = data['user_id']
    recommendations = data['recommendations']
    timestamp = data.get('timestamp')  # Ambil timestamp dari request

    # Pastikan timestamp ada
    if not timestamp:
        return jsonify({'error': 'Timestamp is required'}), 400

    # Mengonversi timestamp ke format yang lebih mudah dibaca
    timestamp = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')  # Dari milidetik ke detik

    # Memastikan recommendations adalah list
    if not isinstance(recommendations, list):
        return jsonify({'error': 'Recommendations must be an array'}), 400

    # Validasi setiap rekomendasi
    for scenario in recommendations:
        if 'scenario' not in scenario or 'products' not in scenario:
            return jsonify({'error': 'Each recommendation must include scenario and products'}), 400
        if not isinstance(scenario['products'], list):
            return jsonify({'error': 'Products must be a list'}), 400
        for product in scenario['products']:
            # Validasi tambahan untuk revOrNot dan order
            if 'product_id' not in product or 'revOrNot' not in product or 'order' not in product:
                return jsonify({'error': 'Each product must include product_id, revOrNot, and order'}), 400
            if not isinstance(product['revOrNot'], (bool, type(None))):
                return jsonify({'error': 'revOrNot must be a boolean or null'}), 400
            if not isinstance(product['order'], int) or not (0 <= product['order'] <= 5):
                return jsonify({'error': 'order must be an integer between 0 and 5'}), 400

    # Simpan data ke MongoDB, sertakan timestamp
    try:
        recommendation_data = {
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': timestamp  # Simpan timestamp yang sudah dikonversi
        }
        recommendations_collection_2.insert_one(recommendation_data)
        return jsonify({'message': 'Recommendation saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


from bson import ObjectId

@app.route('/user/<user_id>', methods=['GET'])
def get_user_data(user_id):
    try:
        user_id = int(user_id)  # Ensure the user_id is an integer
    except ValueError:
        return jsonify({"message": "Invalid user ID"}), 400  # Return a 400 if user_id is not a valid integer

    user = user_collection.find_one({"user_id": user_id})  # Use find_one for a single document

    if user:
        # Convert ObjectId to string before returning the response
        user['_id'] = str(user['_id'])  # Convert the _id field to a string
        return jsonify(user), 200  # Return the user data directly
    else:
        return jsonify({"message": "No user found"}), 404  # 404 if no user is found

@app.route('/user', methods=['GET'])
def get_all_user():
    """Fetch all product descriptions from MongoDB."""
    try:
        user = user_collection.find()  # Sort by price in ascending order
        return dumps(user)  # Use dumps for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/products', methods=['GET'])
def get_all_products():
    """Fetch all product descriptions from MongoDB."""
    try:
        # products = collection.find()
        products = collection.find().sort('jumlah_review', 1)  # Sort by price in ascending order
        return dumps(products)  # Use dumps for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/products/popular', methods=['GET'])
def get_popular_products():
    """Fetch all product descriptions from MongoDB."""
    try:
        products = collection.find().sort("price", -1).limit(50)  # Sort by price in ascending order
        return dumps(products)  # Use dumps for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/products/<int:product_id>', methods=['GET'])  # Mengubah tipe parameter ke int
def get_product(product_id):
    """Fetch a product description by ID from MongoDB."""
    try:
        print(f"Received product_id: {product_id}")  # Debugging line
        product = collection.find_one({"product_id": product_id})  # Query menggunakan product_id sebagai Int
        if product:
            print(product) 
            return dumps(product)  # Serialize the product data
        else:
            return jsonify({"error": "Product not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def normalize_ratings(df, column_name):
    min_rating = df[column_name].min() 
    max_rating = df[column_name].max() 
    df.loc[:, column_name] = (df[column_name] - min_rating) / (max_rating - min_rating)
    return df


def top_n_recommendations_unique(recommendations, target_makeup_part, target_makeup_type, top_n):

    if 'final_score' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='final_score', ascending=False)
    elif 'score_svd' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='score_svd', ascending=False)
    elif 'score' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='score', ascending=False)
    else:
         sorted_recommendations = recommendations  # No sorting if neither 'score' nor 'score_svd' exists

    unique_recommendations = sorted_recommendations.drop_duplicates(subset=['product_name'])
    print("Total unique_recommendations:", len(unique_recommendations))

    unique_recommendations_df = unique_recommendations[unique_recommendations['makeup_type'] == target_makeup_type]
    print("Total unique_recommendations_df with sub category:", len(unique_recommendations_df))
    top_n_recommendations_df = unique_recommendations_df.head(top_n)

    return top_n_recommendations_df


file = pd.read_csv('normalization_data.csv')
file.columns = file.columns.str.strip()
file["After"] = file["After"].fillna(file["Before"])
norm_dict = pd.Series(file["After"].values, index=file["Before"]).to_dict()
stemmer = SnowballStemmer("english")

# Fungsi untuk normalisasi teks
def normalize_text(text, norm_dict):
    words = text.split() 
    normalized_words = [norm_dict.get(word, word) for word in words]
    return ' '.join(normalized_words) 


stop_words = set(stopwords.words('english'))
file_path = "list stopword.xlsx"
df = pd.read_excel(file_path)

# Menambahkan stopwords tambahan dari kolom 'Hapus' jika kolom ada
if 'Hapus' in df.columns:
    additional_stopwords = set(df['Hapus'].dropna().astype(str))  # Pastikan konversi ke string
    stop_words.update(additional_stopwords)  # Gabungkan dengan stopwords bawaan
else:
    print("Kolom 'Hapus' tidak ditemukan dalam file Excel.")

def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def preprocess_user_description(description, norm_dict):
    description = description.lower()
    description = description.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    description = " ".join(description.split())
    description = normalize_text(description, norm_dict)
    words = nltk.word_tokenize(description)
    words_without_stopwords = remove_stopwords(words)
    stemmed_words = [stemmer.stem(word) for word in words_without_stopwords]
    return ' '.join(stemmed_words)

def cbf_tfidf(makeup_part_input, product_category, user_id, skin_type='', skin_tone='', under_tone='', user_description='', product_id_refs=""):
    if isinstance(product_id_refs, int):
        product_id_refs = [product_id_refs] 

    product_desc = []
    product_names_to_remove = set() 

    # Ambil deskripsi produk berdasarkan product_id_refs
    for product_id_ref in product_id_refs:
        result = products[products["product_id"] == product_id_ref]["combined_info_fix"]
        if not result.empty:
            product_desc.append(result.values[0])
            # Cari product_name untuk produk referensi dan tambahkan ke set
            product_name = products[products["product_id"] == product_id_ref]["product_name"].values[0]
            product_names_to_remove.add(product_name)
            print(f"Berhasil ditambah produk: {product_name}")
        else:
            product_desc.append(None)  # Jika product_id tidak ditemukan, simpan None

    # Menampilkan hasil
    print("CBF - Deskripsi produk:", product_desc)

    combined_description = f"{makeup_part_input} {product_category} suitable for skin tone {skin_tone} undertone {under_tone} skintype {skin_type} additional info {user_description} reference product {product_desc}"
    combined_description = preprocess_user_description(combined_description, norm_dict)
    print("CBF - Combined Description:", combined_description)
    
    all_items = products['product_id'].unique()
    ref_items = set(product_id_refs)
    recommend_item = [item for item in all_items if item not in ref_items]

    print("CBF - Jumlah produk total:", len(all_items))
    print("CBF - Jumlah produk yang sudah dikurangi reference:", len(recommend_item))

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(products['combined_info_fix'])
    target_tfidf = tfidf.transform([combined_description])
    # print("Jumlah kosa kata (vocabulary):", len(tfidf.vocabulary_))


    cosine_sim = cosine_similarity(target_tfidf, tfidf_matrix)
    similarity_scores = list(enumerate(cosine_sim[0]))
    sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Filter rekomendasi produk yang belum diberi rating dan menghapus produk dengan nama yang sama
    similar_products_filtered = [
        {
            "product_id": int(products['product_id'].iloc[i]),
            "product_name": products['product_name'].iloc[i],
            "makeup_part": products['makeup_part'].iloc[i],
            "makeup_type": products['makeup_type'].iloc[i],
            "shade_name": products['shade_name'].iloc[i],
            "combined_info_fix": products['combined_info_fix'].iloc[i],
            "score": float(score)
        }
        for i, score in sorted_similar_items if products['product_id'].iloc[i] in recommend_item and products['product_name'].iloc[i] not in product_names_to_remove
    ]

    similar_products_filtered_df = pd.DataFrame(similar_products_filtered, columns=['product_id', 'product_name', 'makeup_part', 'makeup_type', 'shade_name', 'combined_info_fix', 'score'])

    print("CBF - Jumlah produk yang namanya tidak sama dengan referensi:", len(similar_products_filtered_df))
    sorted_df = similar_products_filtered_df.sort_values(by='score', ascending=False)

    return sorted_df, makeup_part_input, product_category


@app.route('/recommend/tfidf', methods=['GET'])
def recommend_tfidf():
    # Get parameters from query string
    makeup_part_input = request.args.get('makeup_part_input', default='', type=str)
    product_category = request.args.get('product_category', default='', type=str)
    user_id = request.args.get('user_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str)
    skin_tone = request.args.get('skin_tone', default='', type=str)
    under_tone = request.args.get('under_tone', default='', type=str)
    user_description = request.args.get('user_description', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)  # Added top_n parameter
    product_id_refs = request.args.get('product_id_refs', type=int)

    similar_products, target_makeup_part, target_makeup_type = cbf_tfidf(
        makeup_part_input, product_category, user_id, skin_type, skin_tone, under_tone, user_description,product_id_refs
    )

    recommendations_df = top_n_recommendations_unique(similar_products, target_makeup_part, target_makeup_type, top_n)

    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }

    return jsonify(response)


def svd(makeup_part_input, product_category, user_id):
    model_path = 'svd_model_new_1.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    all_items = products['product_id'].unique()
    user_ratings = data[data['user_id'] == user_id]
    rated_items = user_ratings['product_id'].unique()
    unrated_products = [item for item in all_items if item not in rated_items]
    print('SVD - Produk keseluruhan:',len(all_items))
    print('SVD - Produk yang sudah di rating:',len(rated_items))
    print('SVD - Produk yang belum di rating:',len(unrated_products))

    predictions = []
    for product_id in unrated_products:
        pred = model.predict(user_id, product_id)
        predicted_score = pred.est
        predictions.append((product_id, predicted_score))

    # Buat DataFrame dari hasil prediksi
    predictions_df = pd.DataFrame(predictions, columns=['product_id', 'score_svd'])

    merged_recommendations = predictions_df.merge(
        products[['product_id', 'product_name', 'makeup_part', 'makeup_type', 'shade_name', 'combined_info_fix']],
        on='product_id',
        how='left'
    )
    
    normalized_df = normalize_ratings(merged_recommendations, 'score_svd')
    sorted_df = normalized_df.sort_values(by='score_svd', ascending=False)
    
    return sorted_df, makeup_part_input, product_category
@app.route('/recommend/svd', methods=['GET'])
def recommend_svd():
    user_id = request.args.get('user_id', type=int)
    makeup_part_input = request.args.get('makeup_part_input', default='', type=str)
    product_category = request.args.get('product_category', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)

    recommendations, target_makeup_part, target_makeup_type = svd(
        makeup_part_input, product_category,user_id
    )

    recommendations_df = top_n_recommendations_unique(recommendations, target_makeup_part, target_makeup_type, top_n)

    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }
    return jsonify(response)

def hybrid_tfidf(makeup_part_input, product_category, user_id, 
                 skin_type='', skin_tone='', under_tone='', user_description='',product_id_refs='',
                 cbf_weight=None, cf_weight=None):
   
    # Content-Based Filtering
    similar_products, target_makeup_part, target_makeup_type = cbf_tfidf(
        makeup_part_input, product_category, user_id, skin_type, skin_tone, under_tone, user_description,product_id_refs
    )
    print("HYBRID - CBF done.")

    # Collaborative Filtering
    normalized_df, target_makeup_part, target_makeup_type = svd(makeup_part_input, product_category, user_id)
    print("HYBRID - CF done.")

    print("HYBRID - Length of similar_products:", len(similar_products))
    print("HYBRID - Length of normalized_df:", len(normalized_df))

    # Combine the results from CBF and SVD
    combined_df = pd.merge(similar_products, normalized_df, on='product_id', how='inner')
    # print("Combined DataFrame created.")
    print("HYBRID - Length of combined_df:", len(combined_df))
    print(cbf_weight)
    print(cf_weight)
    total_weight=cf_weight+cbf_weight
    print(total_weight)

    # Apply the combined scoring from CBF and CF
    combined_df['final_score'] = (cbf_weight/total_weight * combined_df['score']) + (cf_weight/total_weight * combined_df['score_svd'])

    # Sort the filtered results by the final score
    combined_df_sorted = combined_df.sort_values(by='final_score', ascending=False)
    combined_df_sorted = combined_df_sorted.drop(columns=['product_name_y', 'makeup_part_y', 'makeup_type_y', 'shade_name_y', 'combined_info_fix_y'])
    combined_df_sorted = combined_df_sorted.rename(columns={'product_name_x': 'product_name', 'makeup_part_x': 'makeup_part', 'makeup_type_x': 'makeup_type', 'shade_name_x':'shade_name', 'combined_info_fix_x':'combined_info_fix'})

    return combined_df_sorted, makeup_part_input, product_category

@app.route('/recommend/hybrid_tfidf', methods=['GET'])
def recommend_hybrid_tfidf():
    makeup_part_input = request.args.get('makeup_part_input', default='', type=str)
    product_category = request.args.get('product_category', default='', type=str)
    user_id = request.args.get('user_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str)
    skin_tone = request.args.get('skin_tone', default='', type=str)
    under_tone = request.args.get('under_tone', default='', type=str)
    user_description = request.args.get('user_description', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)
    cbf_weight = request.args.get('cbf_weight', default=None, type=float)
    cf_weight = request.args.get('cf_weight', default=None, type=float)
    product_id_refs = request.args.get('product_id_refs', type=int)

    recommendations_df, target_makeup_part, target_makeup_type = hybrid_tfidf(
        makeup_part_input, product_category, user_id, 
        skin_type, skin_tone, under_tone, user_description,product_id_refs,
        cbf_weight=cbf_weight, cf_weight=cf_weight
    )

    recommendations_df = top_n_recommendations_unique(recommendations_df, target_makeup_part, target_makeup_type, top_n)
  
    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }

    return jsonify(response)



def svd2(makeup_part_input, product_category, user_id):
    model_path = 'svd_model_new_3.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    all_items = products['product_id'].unique()
    user_ratings = data[data['user_id'] == user_id]
    rated_items = user_ratings['product_id'].unique()
    unrated_products = [item for item in all_items if item not in rated_items]
    print('SVD - Produk keseluruhan:',len(all_items))
    print('SVD - Produk yang sudah di rating:',len(rated_items))
    print('SVD - Produk yang belum di rating:',len(unrated_products))

    predictions = []
    for product_id in unrated_products:
        pred = model.predict(user_id, product_id)
        predicted_score = pred.est
        predictions.append((product_id, predicted_score))

    # Buat DataFrame dari hasil prediksi
    predictions_df = pd.DataFrame(predictions, columns=['product_id', 'score_svd'])

    merged_recommendations = predictions_df.merge(
        products[['product_id', 'product_name', 'makeup_part', 'makeup_type', 'shade_name', 'combined_info_fix']],
        on='product_id',
        how='left'
    )
    
    normalized_df = normalize_ratings(merged_recommendations, 'score_svd')
    sorted_df = normalized_df.sort_values(by='score_svd', ascending=False)
    
    return sorted_df, makeup_part_input, product_category
@app.route('/recommend/svd2', methods=['GET'])
def recommend_svd2():
    user_id = request.args.get('user_id', type=int)
    makeup_part_input = request.args.get('makeup_part_input', default='', type=str)
    product_category = request.args.get('product_category', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)

    recommendations, target_makeup_part, target_makeup_type = svd2(
        makeup_part_input, product_category,user_id
    )

    recommendations_df = top_n_recommendations_unique(recommendations, target_makeup_part, target_makeup_type, top_n)

    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }
    return jsonify(response)

def hybrid_tfidf2(makeup_part_input, product_category, user_id, 
                 skin_type='', skin_tone='', under_tone='', user_description='',product_id_refs='',
                 cbf_weight=None, cf_weight=None):
   
    # Content-Based Filtering
    similar_products, target_makeup_part, target_makeup_type = cbf_tfidf(
        makeup_part_input, product_category, user_id, skin_type, skin_tone, under_tone, user_description,product_id_refs
    )
    print("HYBRID - CBF done.")

    # Collaborative Filtering
    normalized_df, target_makeup_part, target_makeup_type = svd2(makeup_part_input, product_category, user_id)
    print("HYBRID - CF done.")

    print("HYBRID - Length of similar_products:", len(similar_products))
    print("HYBRID - Length of normalized_df:", len(normalized_df))

    # Combine the results from CBF and SVD
    combined_df = pd.merge(similar_products, normalized_df, on='product_id', how='inner')
    # print("Combined DataFrame created.")
    print("HYBRID - Length of combined_df:", len(combined_df))
    print(cbf_weight)
    print(cf_weight)
    total_weight=cf_weight+cbf_weight
    print(total_weight)

    # Apply the combined scoring from CBF and CF
    combined_df['final_score'] = (cbf_weight/total_weight * combined_df['score']) + (cf_weight/total_weight * combined_df['score_svd'])

    # Sort the filtered results by the final score
    combined_df_sorted = combined_df.sort_values(by='final_score', ascending=False)
    combined_df_sorted = combined_df_sorted.drop(columns=['product_name_y', 'makeup_part_y', 'makeup_type_y', 'shade_name_y', 'combined_info_fix_y'])
    combined_df_sorted = combined_df_sorted.rename(columns={'product_name_x': 'product_name', 'makeup_part_x': 'makeup_part', 'makeup_type_x': 'makeup_type', 'shade_name_x':'shade_name', 'combined_info_fix_x':'combined_info_fix'})

    return combined_df_sorted, makeup_part_input, product_category

@app.route('/recommend/hybrid_tfidf2', methods=['GET'])
def recommend_hybrid_tfidf2():
    makeup_part_input = request.args.get('makeup_part_input', default='', type=str)
    product_category = request.args.get('product_category', default='', type=str)
    user_id = request.args.get('user_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str)
    skin_tone = request.args.get('skin_tone', default='', type=str)
    under_tone = request.args.get('under_tone', default='', type=str)
    user_description = request.args.get('user_description', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)
    cbf_weight = request.args.get('cbf_weight', default=None, type=float)
    cf_weight = request.args.get('cf_weight', default=None, type=float)
    product_id_refs = request.args.get('product_id_refs', type=int)

    recommendations_df, target_makeup_part, target_makeup_type = hybrid_tfidf2(
        makeup_part_input, product_category, user_id, 
        skin_type, skin_tone, under_tone, user_description,product_id_refs,
        cbf_weight=cbf_weight, cf_weight=cf_weight
    )

    recommendations_df = top_n_recommendations_unique(recommendations_df, target_makeup_part, target_makeup_type, top_n)
  
    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }

    return jsonify(response)



def calculate_score(data):
    correct_order = [1, 2, 3, 4, 5]  # Urutan ideal
    scores = []

    recommended_orders = [item["order"] for item in data]

    for i, order in enumerate(recommended_orders):  
        try:
            correct_pos = correct_order.index(order)
            if correct_pos == i:
                scores.append(3)
            elif abs(correct_pos - i) == 1:
                scores.append(2)
            else:
                scores.append(1)
        except ValueError:
            scores.append(0)

    return scores

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    denominators = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2**relevance_scores - 1) / denominators)
    return dcg

# def ndcg_for_all(relevance_scores):
#     k = len(relevance_scores)
#     dcg = dcg_at_k(relevance_scores, k)
#     ideal_relevance_scores = sorted(relevance_scores, reverse=True)
#     idcg = dcg_at_k(ideal_relevance_scores, k)
#     ndcg = dcg / idcg* 100 if idcg > 0 else 0
#     return dcg, idcg, ndcg

# def calculate_precision(data):
#     relevant_items = sum(1 for item in data if item["revOrNot"])
#     precision = relevant_items / len(data) *100 if data else 0
#     return precision

def ndcg_for_all(relevance_scores):
    k = len(relevance_scores)
    dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance_scores, k)
    ndcg = (dcg / idcg * 100) if idcg > 0 else 0
    
    # Membulatkan ke dua angka desimal
    return round(dcg, 2), round(idcg, 2), round(ndcg, 2)

def calculate_precision(data):
    relevant_items = sum(1 for item in data if item["revOrNot"])
    precision = (relevant_items / len(data) * 100) if data else 0
    
    # Membulatkan ke dua angka desimal
    return round(precision, 2)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    request_data = request.json

    if not request_data or "recommendations" not in request_data:
        return jsonify({"error": "Invalid input format"}), 400

    user_id = request_data.get("user_id", None)
    recommendations = request_data["recommendations"]

    results = []

    for scenario_data in recommendations:
        scenario = scenario_data["scenario"]
        products = scenario_data["products"]

        # Hitung skor
        scores = calculate_score(products)

        # Hitung NDCG
        dcg, idcg, ndcg = ndcg_for_all(scores)

        # Hitung Precision
        precision = calculate_precision(products)

        results.append({
            "scenario": scenario,
            "dcg": dcg,
            "idcg": idcg,
            "ndcg": ndcg,
            "precision": precision
        })

    response = {
        "user_id": user_id,
        "results": results
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
CORS(app)
