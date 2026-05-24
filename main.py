from rag.generator import Generator


def main():
    generator = Generator()
    print(
        "🧠 Mental Health RAG Chatbot  (type 'quit' to exit, 'top_k=N' to change retrieval depth)\n"
    )
    top_k = 3

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye. Take care of yourself 💙")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye. Take care of yourself 💙")
            break

        # allow runtime top_k change: top_k=5
        if user_input.lower().startswith("top_k="):
            try:
                top_k = int(user_input.split("=")[1])
                print(f"  ✅ top_k set to {top_k}")
            except ValueError:
                print("  ❌ Usage: top_k=<integer>")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            reply = generator.answer(user_input, top_k=top_k, verbose=False)
            print(reply)
        except Exception as e:
            print(f"[Error] {e}")
        print()


if __name__ == "__main__":
    main()
