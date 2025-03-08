from flask import Flask, request, jsonify

app = Flask(__name__)

data = {"name": "John", "age": 30}


@app.route("/greet/<username>", methods=["GET"])
def greet(username):
    if not username:
        return jsonify({"error": "Username is required"}), 400
    return f"Hello, {username}! Welcome to our service."

@app.route("/echo", methods=["POST"])
def echo():
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        return jsonify({"message": "Received data", "data": data}), 201

@app.route("/update_age/<int:user_id>", methods=["PUT"])
def update_age(user_id):
    if request.method == "PUT":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        new_age = data.get("age")
        if new_age is None:
            return jsonify({"error": "Age is required"}), 400
        return jsonify({"message": f"User {user_id} age updated to {new_age}"}), 200

@app.route("/delete_user/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    if request.method == "DELETE":
        # Here you would typically delete the user from the database
        # For now, we'll just return a success message
        return jsonify({"message": f"User {user_id} has been deleted"}), 200

@app.route("/4071")
def index1():
    return "Hello Paal"

@app.route("/4072")
def index2():
    return "Hello Zoker", 201

if __name__ == '__main__':
    app.run(port=3000, debug=True)