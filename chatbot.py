"""
CLI entrypoint: pick a demo user (role) -> chat loop with RBAC + guardrails applied.
"""

from rbac import DEMO_USERS
from rag_chain import retrieve_with_scores, generate_answer
from guardrails.scope import is_out_of_scope, OUT_OF_SCOPE_MESSAGE
from guardrails.pii import redact_pii


def choose_user() -> dict:
    print("FinSolve Technologies -- Internal Assistant (demo login)\n")
    usernames = list(DEMO_USERS.keys())
    for i, username in enumerate(usernames, start=1):
        user = DEMO_USERS[username]
        print(f"  {i}. {user['name']} -- {user['title']} (role: {user['role']})")

    while True:
        choice = input(f"\nLog in as [1-{len(usernames)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(usernames):
            return DEMO_USERS[usernames[int(choice) - 1]]
        print("Invalid choice, try again.")


def chat_loop(user: dict):
    role = user["role"]
    print(f"\nLogged in as {user['name']} ({user['title']}). Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        results = retrieve_with_scores(question, role)

        if is_out_of_scope(results):
            print(f"\nAssistant: {OUT_OF_SCOPE_MESSAGE}\n")
            continue

        docs = [doc for doc, _ in results]
        try:
            answer = generate_answer(question, docs)
        except RuntimeError as e:
            print(f"\nAssistant: [Cannot generate an answer -- {e}]\n")
            continue

        answer = redact_pii(answer, role)
        sources = sorted({doc.metadata.get("source") for doc in docs})

        print(f"\nAssistant: {answer}")
        print(f"Sources: {', '.join(sources)}\n")


def main():
    user = choose_user()
    chat_loop(user)


if __name__ == "__main__":
    main()
