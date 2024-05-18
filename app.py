from flask import Flask, render_template, request, send_file, jsonify, abort, redirect, url_for
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import io
import csv


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


app = Flask(__name__)

model = joblib.load('svc_pipeline.pkl')

# Define the mapping of predicted values to categories
price_categories = {
    0: 'Low Cost',
    1: 'Medium Cost',
    2: 'High Cost',
    3: 'Very High Cost'
}

# In-memory database for devices
devices = []
device_id_counter = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    predictions = model.predict(df[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'm_dep',
                                    'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'talk_time', 'three_g',
                                    'touch_screen', 'wifi']])

    # Map the predicted values to categories
    predicted_categories = [price_categories[p] for p in predictions]

    df['Predicted Price Range'] = predicted_categories

    # Convert DataFrame to CSV
    csv_data = df.to_csv(index=False)

    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )


@app.route('/api/devices', methods=['GET'])
def get_devices():
    devices = []
    with open('train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            devices.append(row)
    print(devices)
    return jsonify(devices)


@app.route('/api/get_device/<int:device_id>', methods=['GET'])
def get_device(device_id):
    with open('predictions.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['id']) == device_id:
                return jsonify(row)
    return jsonify({'error': 'Device not found'}), 404


# Endpoint to add a new device
@app.route('/add_device', methods=['POST'])
def add_device():
    data = {
        'battery_power': request.form['battery_power'],
        'blue': request.form['blue'],
        'clock_speed': request.form['clock_speed'],
        'dual_sim': request.form['dual_sim'],
        'fc': request.form['fc'],
        'four_g': request.form['four_g'],
        'm_dep': request.form['m_dep'],
        'n_cores': request.form['n_cores'],
        'pc': request.form['pc'],
        'px_height': request.form['px_height'],
        'px_width': request.form['px_width'],
        'ram': request.form['ram'],
        'sc_h': request.form['sc_h'],
        'talk_time': request.form['talk_time'],
        'three_g': request.form['three_g'],
        'touch_screen': request.form['touch_screen'],
        'wifi': request.form['wifi'],

    }

    with open('test.csv', 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)

    return redirect(url_for('device_added'))


@app.route('/device_added')
def device_added():
    return "Device added successfully!"


if __name__ == '__main__':
    app.run(debug=False, )

