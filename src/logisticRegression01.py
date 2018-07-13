import numpy as np

feature_size = 10
num_examples = 300
output_classes = 2
alpha = 0.001
num_iterations = 1000
w = np.zeros((output_classes, feature_size))
b = np.zeros((output_classes, 1))

x = np.random.normal(0.5, 0.5, (feature_size, num_examples))
y = [np.sum(x, 0), np.sum(-x, 0)]
#y = np.max(x, 0)
yt = np.reshape(y, (output_classes, num_examples))
for num_iter in range(num_iterations):
    z = np.dot(w, x)
    a = 1/(1 + np.exp(-z))
    J = (-np.vdot(yt, np.log(a)) - np.vdot(1-yt, np.log(1-a)))/num_examples
    dz = a - yt
    db = np.reshape(np.sum(dz, 1)/num_examples, (output_classes, 1))
    dw = np.transpose(np.matmul(x, np.transpose(dz))) / num_examples
    w = w - alpha * dw
    b = b - alpha * db
    print(num_iter, 'iteration', J)

print('W', w)
print('B', b)



