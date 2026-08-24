from controller import Controller


def main():
    controller = Controller()
    print("Multi-Agent Forecasting System started.")

    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = controller.run(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()