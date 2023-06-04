from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the saved TensorFlow model
model = tf.keras.models.load_model('./models/rottenvsfresh.h5')

@app.route('/api/predict', methods=['POST'])
def predict_fruit():
    try:
        data = request.get_json()
        image_array = np.array(data['image'])
        image_array = image_array.reshape(1, 100, 100, 3)

        prediction = model.predict(image_array)[0][0]

        if prediction != 1:
            result = {'class': 'Fresh fruit', 'confidence': float(prediction)}
        else:
            result = {'class': 'Rotten fruit', 'confidence': float(prediction)}

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
