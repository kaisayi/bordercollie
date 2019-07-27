# 感知机公式推导

- [感知机公式推导](#%e6%84%9f%e7%9f%a5%e6%9c%ba%e5%85%ac%e5%bc%8f%e6%8e%a8%e5%af%bc)
  - [1.感知机](#1%e6%84%9f%e7%9f%a5%e6%9c%ba)
    - [1.1.基本架构](#11%e5%9f%ba%e6%9c%ac%e6%9e%b6%e6%9e%84)
    - [1.2.链式法则](#12%e9%93%be%e5%bc%8f%e6%b3%95%e5%88%99)
    - [1.3.多层感知机传递推导](#13%e5%a4%9a%e5%b1%82%e6%84%9f%e7%9f%a5%e6%9c%ba%e4%bc%a0%e9%80%92%e6%8e%a8%e5%af%bc)
    - [1.4.`Himmelblau function`优化](#14himmelblau-function%e4%bc%98%e5%8c%96)

## 1.感知机

### 1.1.基本架构

![forward](https://blog-oss-youzz.oss-cn-shanghai.aliyuncs.com/forward_1563985257489.jpg)

其中:

$$
\begin{aligned}
    &E = \frac{1}{2} (O^1_j - t_j)^2 \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t_j) \frac{\partial O^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t_j) \sigma (x^1_j)(1 - \sigma (x^1_j)) \frac{\partial x^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t_j) O^1_j (1 - O^1_j) \frac{\partial x^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t_j) O^1_j (1 - O^1_j) x^0_i
\end{aligned}
$$

代码:

```python
import tensorflow as tf

x = tf.random.normal([1, 3])
w = tf.ones([3, 1])
b = tf.ones([1])
y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.sigmoid(x@w + b)
    loss = tf.reduce_mean(tf.losses.MSE(y, prob))

grads = tape.gradient(loss, [w, b])
print("Gradient of w: ", grads[0])
print("Gradient of b: ", grads[1])

# Gradient of w:  tf.Tensor(
# [[-0.01003827]
#  [ 0.0039798 ]
#  [-0.00257523]], shape=(3, 1), dtype=float32)
# Gradient of b:  tf.Tensor([-0.00432863], shape=(1,), dtype=float32)
```

### 1.2.链式法则

基本原理：

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \frac{\partial u}{\partial x}
$$

example:

$$
\begin{aligned}
    y_2 &= y_1 w_2 + b_2\\
    y_1 &= x w_1 + b_1\\
    \frac{\partial y_2}{\partial w_1}?
\end{aligned}
$$

代码实现：

```python
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, b1, w2, b2])

    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]

print("dy2_dy1: ", dy2_dy1)
print("dy1_dw1: ", dy1_dw1)
print("dy2_dw1: ", dy2_dw1)
```

### 1.3.多层感知机传递推导

![chain-rule](https://blog-oss-youzz.oss-cn-shanghai.aliyuncs.com/chain-rule_1564069195801.jpg)

$$
\begin{aligned}
    E &= \frac{1}{2} \sum_{m \in M} (O_m^t - y_m)^2\\
    \frac{\partial E}{\partial W_{ij}^{t-1}} &= \sum_{m \in M} (O_m^t - y_m) \frac{\partial}{\partial W_{ij}^{t-1}} \sigma (x_m^t)\\
    &= \sum_{m \in M} (O_m^t - y_m) \sigma (x_m^t) (1 - \sigma (x_m^t)) \frac{\partial x_m^t}{\partial W_{ij}^{t-1}}\\
    &= \sum_{m \in M} (O_m^t - y_m) O_m^t (1 - O_m^t) W_{jm}^t \frac{\partial O_j^{t-1}}{\partial W_{ij}^{t-1}}\\
    &= \frac{\partial O_j^{t-1}}{\partial W_{ij}^{t-1}} \sum_{m \in M} (O_m^t - y_m) O_m^t (1 - O_m^t) W_{jm}^t\\
    &= O_j^{t-1}(1 - O_j^{t-1}) \frac{\partial x_j^{t-1}}{\partial W_{ij}^{t-1}} \sum_{m \in M} (O_m^t - y_m) O_m^t (1 - O_m^t) W_{jm}^t\\
    &= O_i^{t-2} O_j^{t-1}(1 - O_j^{t-1}) \sum_{m \in M} (O_m^t - y_m) O_m^t (1 - O_m^t) W_{jm}^t
\end{aligned}
$$

从上式可以看出：

$$
\frac{\partial E}{\partial W_{jk}^{t}} = O_j^{t-1} \delta_k^t
$$

其中，$O_j^{t-1}$ 为当前层的输入；$\delta_k^t$ 为从误差传递积累的信息

对于最后一层:

$$
\delta_k^{-1} = O_k^{-1} (1 - O_k^{-1}) f(O_k^{-1}, y_k)
$$

对于中间层：

$$
\delta_j^{l} = O_j^l (1 - O_j^l) \sum_{m \in M} \delta_m^{l+1} W_{jm}^{l+1}
$$

### 1.4.`Himmelblau function`优化

$$
f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2
$$

代码：

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def visualize():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print('x, y range: ', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X, Y range: ', X.shape, Y.shape)
    Z = himmelblau([X, Y])

    fig = plt.figure('Himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def train_op():
    x = tf.constant([-4., 0.])

    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads

        if step % 20 == 0:
            print('step {} : x = {}, f(x) = {}'
                  .format(step, x.numpy(), y.numpy()))


if __name__ == "__main__":
    visualize()
```
