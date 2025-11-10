# # app.py
# from flask import Flask, request, jsonify
# from qa import answer

# app = Flask(__name__)

# # @app.route("/ask", methods=["GET"])
# @app.route("/", methods=["GET"])
# def ask():
#     q = request.args.get("q") or request.args.get("question") or ""
#     if not q:
#         return jsonify({"error":"provide ?q=your question"}), 400
#     a = answer(q)
#     return jsonify({"answer": a})

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status":"ok"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)



# app.py
from qa import answer

def main():
    print("=== QA Console App ===")
    print("Type your question below. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = answer(user_input)
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
