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

---

### Simplified Cost Function and Gradient Descent 
We can compress our cost function's two conditional cases into one case:
$$
Cost(h_\theta(x),y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))
$$
We can fully write out our entire cost function as follows:
$$
J(\theta)=-\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]
$$
A vectorized implementation is:
向量化的实现是:
$$
h=g(X\theta)
$$
$$
J(\theta)=\frac{1}{m}·(-y^Tlog(h)-(1-y)^Tlog(1-h))
$$
**Gradient Descent**
Remember that the general form of gradient descent is:

***Repeat***{
 $\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)$
}
We can work out the derivative part using calculus to get:
***Repeat***{
    $\theta_j:=\theta_j-\frac{\alpha}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$
}

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in $\theta$.
A vectorized implementation is:
$$
\theta := \theta - \frac{\alpha}{m}X^T(g(X\theta)-\vec{y})
$$

**add: 逻辑斯蒂函数 h(z) 求导：**
$$
h'(z)=h(z)(1-h(z))
$$
![J求导](https://img.imgdb.cn/item/60434ee7360785be54a5231f.jpg)

---

### Advanced Optimization
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent.  We suggest that you should not write these more sophisticated algorithms yourself but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value θ:
&emsp;&emsp;&emsp;$J(\theta)$
&emsp;&emsp;&emsp;$\frac{\partial}{\partial\theta_j}J(\theta)$
We can write a single function that returns both of these:
```matlab
function [jVal, gradient] = costFunction(theta)
    jVal = [...code to compute J(theta)]
    gradient = [...code to compute derivative of J(theta)]
end
```
Then we can use octave's **"fminunc()"** optimization algorithm along with the **"optimset()"** function that creates an object containing the options we want to send to "fminunc()". 
```matlab
options = optimset('GradObj', 'on', 'MaxIter', 400);
initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

>Specifically, we set the GradObj option to on, which tells fminunc that our
function returns both the cost and the gradient. This allows fminunc to
use the gradient when minimizing the function. Furthermore, we set the
MaxIter option to 400, so that fminunc will run for at most 400 steps before
it terminates.

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.



---

### The Problem of Overfitting
Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.
欠拟合，或是高偏差，就是我们的假设函数`h`的形状与数据的趋势不匹配，它通常是由函数太简单或使用的属性太少而引起的。 另一个极端——过度拟合，(或高方差?),是由于假设函数拟合了现有的可用训练数据但是不能很好地概括预测新测试数据，这通常是由使用了复杂的函数造成的，产生了许多与数据无关的不必要曲线和角度。
This terminology(术语) is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:
1. Reduce the number of features;
   - Manually select which features to keep.
   - Use a model selection algorithm(studied later in the course).
2. Regularization
   - Keep all the features, but reduce the magnitude(大小) of parameters $\theta_j$
   - Regularization works well when we have a lot of slightly useful features.

---

### Cost Function (add Regularization)
If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.
如果在我们的假设函数上发生了过度拟合，我们可以通过增加函数中一些项的损失来减少它们的权重。
Say we wanted to make the following function more quadratic(adj. [数] 二次的;   n. 二次方程式):
如果说我们想使下面的函数更像二次函数:
$\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$
We'll want to eliminate(忽略) the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:
我们想忽略$\theta_3x^3$ 和 $\theta_4x^4$这两项的影响，除了丢弃这两项或是改变假设函数的形式，我们还可以修改损失函数:
$min_\theta\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2+1000·\theta_3^2+1000·\theta_4^2$
We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero. This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms $\theta_3x^3$ and  $\theta_4x^4$.
![after_Regularization](https://img.imgdb.cn/item/60459d86cef1ec5e6f3dbdc2.jpg)

We could also regularize all of our theta parameters in a single summation as:
**$min_\theta\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^m\theta_j^2$**

The $\lambda$, is the **regularization parameter**. It determines how much the costs of our $\theta$ parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting.
使用上面有加上了额外项的损失函数，我们可以平滑假设函数的输出来减轻过度拟合。

---

### Regularized Linear Regression —— 线性回归的正则化
#### Gradient Descent
We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.
**repeat**{
&emsp;&emsp;$\theta_0:=\theta_0-\alpha\frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_0$
&emsp;&emsp;$\theta_j:=\theta_j-\alpha[(\frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\frac{\lambda}{m}\theta_j]$
}
The term $\frac{\lambda}{m}\theta_j$ performs our regularization. With some manipulation our update rule can also be represented as:
$\frac{\lambda}{m}\theta_j$这一项就表示了我们的正则化。经过一些数学变形，我们的更新规则也可以写成：
&emsp;&emsp;**$\theta_j:=\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{\lambda}{m}\sum^m_{(i=1)}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$**
The first term in the above equation,$1-\alpha\frac{\lambda}{m}$will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update. Notice that the second term is now exactly the same as it was before.
上述方程的第一项，$1-\alpha\frac{\lambda}{m}$总是小于一的。直观看来，$\theta_j$的值在每次更新中都有一定量的减少。注意，方程的第二项现在实际上和以前的形式是一样的。
#### Normal Equation——正则方程
Now let's approach regularization using the alternate method of the non-iterative normal equation.
To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses(括号):
现在我们通过不需要迭代的方法——**normal equation(正则方程)**来实现正则化。
为了加入正则化，normal equation 几乎和以前一样，但是我们需要在括号里再加入一项。
$\theta=(X^TX+\lambda·L)^{-1}X^Ty$
where $L=\left[\begin{matrix} 0 &&&&&\\&1&&&&\\ &&1&&\\ &&& \ddots \\ &&&&1 \end{matrix} \right]$

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number λ.
Recall that if m < n, then $X^TX$ is non-invertible. However, when we add the term $λ⋅L$, then $X^TX + λ⋅L$ becomes invertible.
回忆一下，如果 m < n的话，$X^TX$就是不可逆的。然而，当我们加入了$λ⋅L$这一项的话，那$X^TX + λ⋅L$就是可逆的。

---

### Regularized Logistic Regression —— 逻辑回归的正则化
We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting.
我们可以用和正则化线性回归相似的方法来正则化逻辑回归。所以，我们也能避免过度拟合。
**Cost Function**
We can regularize this equation by adding a term to the end:
**$J(\theta)=-\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j$**
The second sum, $\sum^n_{j=1}\theta^2_j$ **means to explicitly exclude**  the bias term, $\theta_0$.
第二项和——$\sum^n_{j=1}\theta^2_j$，明确说明了不包含偏倚项， $\theta_0$.
we should continuously update the two following equations:
**repeat**{
&emsp;&emsp;$\theta_0:=\theta_0-\alpha\frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_0$
&emsp;&emsp;$\theta_j:=\theta_j-\alpha[(\frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\frac{\lambda}{m}\theta_j]$
}



