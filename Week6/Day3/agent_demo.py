def calculator(expression):
    return eval(expression)

def web_search(query):
    return f"Search Result: Information about '{query}'"

print("Task 1")
print(web_search("LangChain"))

print("\nTask 2")
print(calculator("15*8"))

print("\nTask 3")
print(web_search("Artificial Intelligence"))