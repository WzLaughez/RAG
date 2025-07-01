from flask import Flask, render_template, request, jsonify, session
from backend.core import run_llm
from flask_session import Session

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/')
def index():
    # Inisialisasi session jika belum ada
    if "user_prompt_history" not in session:
        session["user_prompt_history"] = []
        session["chat_answer_history"] = []
    return render_template('index.html')

@app.route('/get_answer', methods=['GET'])
def get_answer():
    prompt = request.args.get('prompt')
    if prompt:
        generated_response = run_llm(prompt, llm_model_name="llama3")
        formatted_answer = f"{generated_response['answer']} \n\n {generated_response['sources']} "

        session["user_prompt_history"].append(prompt)
        session["chat_answer_history"].append(formatted_answer)
        session.modified = True

        history = list(zip(session["user_prompt_history"], session["chat_answer_history"]))
        return jsonify({"answer": formatted_answer, "history": history})

    return jsonify({"answer": "", "history": []})

if __name__ == "__main__":
    app.run(debug=False)
