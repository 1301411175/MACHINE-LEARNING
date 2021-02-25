# 第三周
---
### Classification
The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values.
分类问题就像回归问题，不同的是我们现在想要预测的值只有少量的离散值。
> Logistic Regression 解决的不是回归问题，而是分类问题。
---
### Hypothesis Representation——Logistic Function
Intuitively, (in classification problem) it also doesn’t make sense for $h_{\theta}$ to take values larger than 1 or smaller than $\theta$ when we know that $y \in {0, 1}$. To fix this, let’s change the form for our hypotheses $h_{\theta}(x)$ to satisfy $0 \leq h_{\theta}(x) \leq 1$. This is accomplished by plugging $\theta^Tx$ into the **Logistic Function**.

直观来说，(在分类问题上) $h_{\theta}$的取值大于1或小于0是没有意义的，因为我们知道$y$的取值范围可表示为 $(y \in {0, 1})$. 为了解决这个问题，我们改变假设函数$h_{\theta}$的形式，以满足$0 \leq h_{\theta}(x) \leq 1$. 这是通过将$\theta^Tx$放入**逻辑斯蒂函数**来实现的。

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

> $h_\theta(x)=g(\theta^Tx)$
> $z=\theta^Tx$
> $g(z)=\frac{1}{1+e^{-z}}$

The following image shows us what the sigmoid function looks like: 
![sigmoid function](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png?expiry=1614124800000&hmac=v8WWMQF5KiaBkHn4zxOdSIZxUikoSJxZ5mh79evrUWI)

The function $g(z)$, shown here, maps any real number to the $(0, 1)$ interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.
这里展示的$g(z)$，将所有的实数都映射到了$(0, 1)$区间内，使其可以用于将别的任何函数转换成更适合解决分类问题的函数。
$h_\theta(x)$ will give us the **probability** that our output is 1. 
$h_\theta(x)$ 代表输出为1的概率。

---
### decision boundary
In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
> $h_\theta\geq0.5 \rightarrow y=1$
> $h_\theta< 0.5 \rightarrow y=1$

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:
> $g(z)\geq 0.5$
> when $z \geq 0$

So if our input to g is $\theta^TX$, then that means:
> $h_\theta(x)=g(\theta_Tx)\geq0.5$
> when $\theta_Tx\geq0$

From these statements we can now say:
> $\theta_Tx \geq 0 \Rightarrow y=1$ 
> $\theta_Tx < 0 \Rightarrow y=0$ 

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.
Example:
> $\theta=\begin{bmatrix}{5}\\{-1}\\{0}\end{bmatrix}$
> $y=1$ if $5+(-1)x_1+0x_2\geq0$
> $x_1\leq5$

In this case, our decision boundary is a straight vertical line placed on the graph where $x_1 = 5$, and everything to the left of that denotes $y = 1$, while everything to the right denotes $y = 0$.

---
### Cost Function
We cannot use the same cost function that we use for regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.
我们不能使用与线性回归相同的代价函数，因为Logistic函数会导致输出是波动的，导致许多局部最优。换句话说，它不是一个凸函数(这句话翻译存疑)。

Instead, our cost function for logistic regression looks like:
> $J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})$
> $Cost(h_\theta(x),y)=-log(h_\theta(x))$ &emsp;&emsp;if y=1
> $Cost(h_\theta(x),y)=-log(1-h_\theta(x))$ &emsp;&emsp;if y=0

When y = 1, we get the following plot for $J(\theta)$ vs $h_\theta(x)$:
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1614384000000&hmac=r3P6VUIMTpv-Ja1snNOspUuZu0sfy2e2XiQkLNoG648)
Similarly, when y = 0, we get the following plot for $J(\theta)$ vs $h_\theta(x)$:
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1614384000000&hmac=AW9byzzqCg7LHRVUvDCt070NekhCmDs83TH8z_4jmhY)
>$Cost(h_\theta(x),y)=0 \quad if \quad h_\theta(x)=y$
>$Cost(h_\theta(x),y)\rightarrow\infin\quad if \quad y=0 \quad and \quad h_\theta(x)\rightarrow\ 1$
>$Cost(h_\theta(x),y)\rightarrow\infin\quad if \quad y=1 \quad and \quad h_\theta(x)\rightarrow\ 0$

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.
如果我们正确的答案 'y' 为0，那么如果我们的假设函数也输出0，则代价函数将为0。如果我们的假设函数接近1，则代价函数将接近无穷大。
