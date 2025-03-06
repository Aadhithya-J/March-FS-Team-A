# Open the file in read mode
with open("TL.txt", "r", encoding="utf-8") as file:
    content = file.read()  # Read the entire file content
    print(content)  # Print the content
