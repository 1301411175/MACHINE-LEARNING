# 第一周
### Supervised Learning
Supervised learning problems are categorized into **"regression"** and **"classification"** problems. In a regression problem, we are trying to predict results within a **continuous** output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a **discrete** output. In other words, we are trying to map input variables into discrete categories.   
监督学习问题分为“**回归**”和“**分类**”问题。在回归问题中，我们试图预测一个连续输出的结果，这意味着我们试图将输入变量映射到某个**连续函数**。在一个分类问题中，我们试图在一个离散输出中预测结果。换句话说，我们试图将输入变量映射到**离散**的类别中。

---
### Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.
With unsupervised learning there is no feedback based on the prediction results.  
无监督学习允许我们在几乎不知道或完全不知道结果的情况下处理问题。我们可以从我们不一定知道变量影响的数据中推导出结构。在无监督学习中，没有基于预测结果的反馈。

---
### Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $(\frac{1}{2})$as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the$(\frac{1}{2})$term.

```math
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m{(\bar{y_i}-y_i)}^2=\frac{1}{2m}\sum_{i=1}^m{(h_\theta(x_i)-y_i)^2}
```
我们可以通过使用代价函数来衡量我们的假设函数的准确性。这是对所有假设结果(输入x和实际输出y)的平均值差异(实际上是平均值的一个更有趣的版本)。

这个函数也被称为“平方误差函数”，或“均方误差”。均值减半$(\frac{1}{2})$为了方便计算梯度下降，因为平方函数的导数项将消去$(\frac{1}{2})$

---
### Gradient Descent
The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter **α**, which is called the **learning rate**.  
我们的方法是对代价函数求导(函数的切线)。函数的斜率就是这一点的导数，它会给我们一个移动的方向。我们在代价函数的方向上以最陡的下降来计算。每一步的大小由参数**α**决定，该参数称为**学习速率**。   
At each iteration j, one should simultaneously update the parameters $\theta_1, \theta_2,...,\theta_n$
 . Updating a specific parameter prior to calculating another one on the $j^{(th)}$
iteration would yield to a wrong implementation.  
在每次迭代j中，我们应该同时更新的参数$\theta_1, \theta_2,...,\theta_n$。在$j^{(th)}$上计算另一个参数之前迭代更新一个另一个参数会导致于错误。  
![image](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56.png?expiry=1613001600000&hmac=Ifp8vvctTw2IK7sBX8x1gkexWALIx53MaoRfrvlJRxQ)  
This method looks at every example in the entire training set on every step, and is called **batch gradient descent**.  
该方法在每个步骤的整个训练集中查看每个示例，称**为批量梯度下降**。



