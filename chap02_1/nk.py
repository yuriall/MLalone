def visit_all(input_node):
  now_node = input_node
  while True:
    print(now_node.data)
    now_node = now_node.next
    if now_node.data == None:
      break