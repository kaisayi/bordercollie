# 感知机公式推导

## 单层感知机

### 基本架构

![forward](https://blog-oss-youzz.oss-cn-shanghai.aliyuncs.com/forward_1563985257489.jpg)

其中:

$$
\begin{aligned}
    &E = \frac{1}{2} (O^1_j - t)^2 \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t) \frac{\partial O^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t) \sigma (x^1_j)(1 - \sigma (x^1_j)) \frac{\partial x^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t) O^1_j (1 - O^1_j) \frac{\partial x^1_j}{\partial w^1_{ij}} \\[2ex]
    &\frac{\partial E}{\partial w^1_{ij}} = (O^1_j - t) O^1_j (1 - O^1_j) x^0_i
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
