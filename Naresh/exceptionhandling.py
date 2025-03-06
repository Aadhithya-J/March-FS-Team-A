def divide_numbers(num1, num2):
    try:
        # Attempting division
        result = num1 / num2
    except ZeroDivisionError:
        # Handle division by zero
        print("Error: You cannot divide by zero.")
    except TypeError:
        # Handle wrong data types
        print("Error: Both inputs must be numbers.")
    else:
        # If no exception occurs
        print(f"Result: {result}")
    finally:
        # Always executed, no matter what
        print("Execution completed.")


