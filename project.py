import sys
from agent import agent_executor
from llm import flush_memory

def main():
    print("Terminal online. Type 'exit' or 'quit' to kill this cell loop.")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Loop broken safely.")
                break

            if not user_input.strip():
                continue

            inputs = {"messages": [("user", user_input)]}
            final_message = None

            # Stream the graph execution
            for output in agent_executor.stream(inputs, stream_mode="values"):
                # This continuously updates with the latest state representation
                if "messages" in output and output["messages"]:
                    final_message = output["messages"][-1]

            # Only print the final response after the stream has completely finished
            if final_message:
                print(f"\nAgent: {final_message.content}")
            else:
                print("\nAgent: [No response generated]")

            # Flush context and GPU memory
            flush_memory()

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break


if __name__ == "__main__":
    main()