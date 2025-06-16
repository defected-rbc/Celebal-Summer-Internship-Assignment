class Node:
    def _init_(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def _init_(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def print_list(self):
        if not self.head:
            print("The list is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        if not self.head:
            print("Cannot delete from an empty list.")
            return
        
        if n <= 0:
            print("Invalid index. Index must be a positive integer.")
            return
        
        current = self.head
        length = 0
        
        while current:
            length += 1
            current = current.next
        
        if n > length:
            print(f"Invalid index. List has only {length} nodes.")
            return
        
        if n == 1:
            self.head = self.head.next
            return
        
        current = self.head
        for _ in range(n - 2):
            current = current.next
        
        current.next = current.next.next

# Testing the implementation
ll = LinkedList()
ll.add_node(10)
ll.add_node(20)
ll.add_node(30)
ll.add_node(40)

print("Initial List:")
ll.print_list()

ll.delete_nth_node(2)
print("\nAfter Deleting 2nd Node:")
ll.print_list()

ll.delete_nth_node(10)
