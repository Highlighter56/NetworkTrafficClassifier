import sys

# Ensure the user provided enough arguments
if len(sys.argv) < 3:
    print("Usage: python greet.py <name> <age>")
    sys.exit(1)

# sys.argv captures EVERYTHING as a string. Convert types manually.
name = sys.argv[1]
age = int(sys.argv[2]) 

print(len(sys.argv))
print(f"Hello {name}, you are {age} years old.")
