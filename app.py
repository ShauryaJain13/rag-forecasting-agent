# from controller import Controller
# from orchestration.state import AgentState


# def main():
#     """
#     Entry point for the forecasting copilot.
#     """

#     controller = Controller()

#     print("Forecasting Copilot started.")
#     print("Type 'exit' or 'quit' to stop.\n")

#     while True:

#         user_input = input("You: ").strip()

#         if user_input.lower() in {"exit", "quit"}:
#             print("Goodbye!")
#             break

#         if not user_input:
#             continue

#         state = AgentState(user_input)

#         try:
#             state = controller.run(state)

#             if state.final_response:
#                 print(f"\nAssistant: {state.final_response}\n")
#             else:
#                 print("\nNo final response was generated.\n")

#         except Exception as e:
#             print(f"\nError: {e}\n")


# if __name__ == "__main__":
#     main()

from controller import Controller
from orchestration.state import AgentState


def main():
    """
    Entry point for the forecasting copilot.
    """

    controller = Controller()

    print("Forecasting Copilot started.")
    print("Type 'exit' or 'quit' to stop.")
    # CHANGED: added an "index" command. Nothing in the codebase ever
    # called KnowledgeBase.index_file() before, so rag_agent could never
    # retrieve anything -- this is the missing entry point to actually
    # populate the knowledge base.
    print("Type 'index <filepath>' to add a document (.pdf/.txt/.csv) "
          "to the knowledge base.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower().startswith("index "):
            filepath = user_input[len("index "):].strip()
            try:
                controller.index_document(filepath)
                print(f"\nIndexed '{filepath}' into the knowledge base.\n")
            except Exception as e:
                print(f"\nError indexing '{filepath}': {e}\n")
            continue

        state = AgentState(user_input)

        try:
            state = controller.run(state)

            if state.final_response:
                print(f"\nAssistant: {state.final_response}\n")
            else:
                print("\nNo final response was generated.\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()