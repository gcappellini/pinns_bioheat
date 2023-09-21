import matplotlib.pyplot as plt

# Vectors
y1 = [5, 6, 7]  # 3 components
y2 = [8, 9]    # 2 components
y3 = [10]      # 1 component

# x-coordinates
x1 = [1, 2, 3]
x2 = [2, 3]
x3 = [3]

# Plotting
plt.plot(x1, y1, '-o', label="Vector y1")
plt.plot(x2, y2, '-o', label="Vector y2")
plt.plot(x3, y3, '-o', label="Vector y3")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Vectors of Different Lengths')
plt.legend()
plt.grid(True)

plt.show()

