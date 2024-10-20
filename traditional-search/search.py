from flask import Flask, request, render_template
import json

app = Flask(__name__, static_folder='images')

# Load mock database from JSON file
def load_data():
    with open('catdogDB.json') as f:
        return json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').lower()
    data = load_data()
    results = [img for img in data['images'] if query in img['description'].lower()]
    print(results)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)