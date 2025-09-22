#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
from utils import enroll_user, check_user, get_enrolled_users
import threading


# In[ ]:


from flask import Flask, render_template, request, jsonify
from utils import enroll_user, check_user, get_enrolled_users

app = Flask(__name__)

@app.route("/")
def index():
    users = get_enrolled_users()
    return render_template("index.html", users=users)

@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json()
    name = data.get("name")

    # Prevent duplicate names
    users = get_enrolled_users()
    if name in users:
        return jsonify({"success": False, "message": f"User '{name}' already enrolled."})

    success, message = enroll_user(name)
    return jsonify({"success": success, "message": message})

@app.route("/verify", methods=["POST"])
def verify():
    success, message, best_name, score = check_user()
    return jsonify({"success": success, "message": message})

if __name__ == "__main__":
    app.run(debug=True)

