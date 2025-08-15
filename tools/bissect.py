#!/usr/bin/env python3

import sys

def interactive_binary_search():
    try:
        target_y = float(input("Enter your target result (y_target): "))

        y_tolerance = 0.01
        x_tolerance = 0.01

        print("\nPlease provide two initial data points from your system.")
        x1 = float(input("Enter your FIRST guess (x1): "))
        y1 = float(input(f" -> What was the result for x1={x1}? (y1): "))

        x2 = float(input("Enter your SECOND guess (x2): "))
        y2 = float(input(f" -> What was the result for x2={x2}? (y2): "))

    except ValueError:
        print("Invalid input. Please enter numbers only. Exiting.")
        sys.exit(1)

    if y1 == y2:
        print("Error: The results for both initial guesses are the same.")
        print("Cannot determine a direction for the search. Exiting.")
        sys.exit(1)

    is_increasing = (y2 - y1) / (x2 - x1) > 0 if x1 != x2 else y2 > y1

    lower_bound_x = min(x1, x2)
    upper_bound_x = max(x1, x2)

    if lower_bound_x == x1:
        lower_bound_y = y1
        upper_bound_y = y2
    else:
        lower_bound_y = y2
        upper_bound_y = y1

    print(f"Initial search range for x: [{lower_bound_x}, {upper_bound_x}]")
    print("-" * 40)

    iteration = 1
    while abs(upper_bound_x - lower_bound_x) > x_tolerance:
        next_x = (lower_bound_x + upper_bound_x) / 2

        print(f"\n--- Iteration {iteration} ---")
        print(f"Suggesting next input: {next_x:.6f}")

        try:
            user_input = input(f" -> Enter the result for this input (or 'q' to quit): ")

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting search.")
                break

            current_y = float(user_input)

            if abs(current_y - target_y) <= y_tolerance:
                print(f"The required input is: {next_x:.6f}")
                break

            if is_increasing:
                if current_y < target_y:
                    print(f"'{current_y}' is LESS than target '{target_y}'. Narrowing search range.")
                    lower_bound_x = next_x
                else:
                    print(f"'{current_y}' is GREATER than target '{target_y}'. Narrowing search range.")
                    upper_bound_x = next_x
            else:
                if current_y < target_y:
                    print(f"'{current_y}' is LESS than target '{target_y}'. Narrowing search range.")
                    upper_bound_x = next_x
                else:
                    print(f"'{current_y}' is GREATER than target '{target_y}'. Narrowing search range.")
                    lower_bound_x = next_x

            print(f"New search range for x: [{lower_bound_x:.6f}, {upper_bound_x:.6f}]")
            iteration += 1

        except ValueError:
            print("Invalid input. Please enter a number for the result or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nSearch interrupted by user. Exiting.")
            break

if __name__ == "__main__":
    interactive_binary_search()
