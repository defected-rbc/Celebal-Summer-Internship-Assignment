def lower_triangular(n):
    print("Lower Triangular Pattern:")
    for i in range(1, n + 1):
        print("* " * i)

def upper_triangular(n):
    print("\nUpper Triangular Pattern:")
    for i in range(n, 0, -1):
        print("  " * (n - i) + "* " * i)

def pyramid(n):
    print("\nPyramid Pattern:")
    for i in range(1, n + 1):
        print(" " * (n - i) + "* " * i)

n = int(input("Enter the number of rows for the patterns: "))

lower_triangular(n)
upper_triangular(n)
pyramid(n)
