import ast
list_str = "[None, 10, 20, 50]"
actual_list = ast.literal_eval(list_str)
print(actual_list)
