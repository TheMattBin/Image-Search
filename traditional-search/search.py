from flask import Flask, request, render_template
import json

app = Flask(__name__)

# Load mock database
with open('mock_database.json') as f:
    mock_db = json.load(f)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q').lower()
    results = [img for img in mock_db['images'] if query in img['description'].lower()]
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)