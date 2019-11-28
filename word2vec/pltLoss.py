import matplotlib.pyplot as plt

loss = []
iteration = []

with open('./all_loss', 'r') as f:
    for line in f:
        y, x = line.split(' ', 1)
        loss.append(y)
        x, _ = x.split('\n', 1)
        iteration.append(x)

loss = loss[: 50]
iteration = iteration[: 50]
plt.title("loss")
plt.scatter(iteration, loss)
# plt.plot(iteration, loss)
plt.show()