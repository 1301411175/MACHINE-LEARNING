# 第二周
### Multiple Features
Linear regression with multiple variables is also known as "multivariate linear regression".  
具有多个变量的线性回归也称为“多元线性回归”。
The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:
**repeat until convergence**:{
&emsp;$\theta_0:=\theta_0−\alpha\frac{1}{m}\sum_{i=1}^{m}(h_θ(x^{(i)})−y^{(i)})⋅x^{(i)}_ 0$

&emsp;$\theta_1:=\theta_1−\alpha\frac{1}{m}\sum_{i=1}^{m}(h_θ(x^{(i)})−y^{(i)})⋅x^{(i)}_ 1$

&emsp;$\theta_2:=\theta_2−\alpha\frac{1}{m}\sum_{i=1}^{m}(h_θ(x^{(i)})−y^{(i)})⋅x^{(i)}_ 2$  

&emsp;...   
}
**In other words:**
**repeat until convergence:**{
&emsp;$\theta_j:=\theta_j−\alpha\frac{1}{m}\sum_{i=1}^{m}(h_θ(x^{(i)})−y^{(i)})⋅x^{(i)}_ j$&emsp;for j:=0...n

}

---
### Feature Scaling(特征缩放)
We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.  
Two techniques to help with this are **feature scaling** and **mean normalization**.Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.  
To implement both of these techniques, adjust your input values as shown in this formula:

```math
x_i:=\frac{x_i-\mu_i}{s_i}
```
Where $\mu_i$ is the **average** of all the values for feature (i) and $s_i$ is the range of values (max - min), or $s_i$ is the standard deviation.Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

我们可以通过让每个输入值在大致相同的范围内来加速梯度下降。这是因为θ会在小范围内快速下降，而在大范围内缓慢下降，因此当变量非常不均匀时，θ会低效率地振荡至最佳。 
有两种技术可以帮助实现这一点，即**特征缩放**和**均值归一化**。**特征缩放**包括用输入值除以输入变量的范围(即最大值减去最小值)，从而得到一个新的范围为1。**均值标准化**包括用输入变量的值减去输入变量的平均值，从而得到输入变量的新平均值为零。为了实现这两种技术，调整你的输入值如下公式所示:
```math
x_i:=\frac{x_i-\mu_i}{s_i}
``` 
$\mu_i$是属性i的均值(max-min)，$s_i$是属性i的标准差。注意分母为范围和标准差会得到不同的结果。本课程的测验使用范围-编程练习使用标准差。

---
### Learning Rate
To summarize:
&emsp;If $\alpha$ is too small: slow convergence. 
&emsp;If $\alpha$ is too large: may not decrease on every iteration and thus may not converge.

---
### Normal Equation
In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the $\theta_j$ ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below: 
在“正规方程”方法中，我们将通过显式地针对取$\theta_j$ 导数并将其设为零来最小化$\theta_j$ 。这使我们无需迭代即可找到最佳$\theta$ 。正规方程公式如下：
```math
\theta=(X^TX)^{-1}X^Ty
``` 
**推导过程：**
&emsp;&emsp;$X^T(y-X\theta)=0$
&emsp;&emsp;$X^Ty=X^TX\theta$
&emsp;&emsp;$\theta=(X^TX)^{-1}X^Ty$
**或**
&emsp;&emsp;$\theta=argmin(y-X\theta)^T(y-X\theta)$
&emsp;&emsp;令 $E=(y-X\theta)^T(y-X\theta)$, 对$\theta$求导得到，
&emsp;&emsp;$\frac{\partial E}{\partial \theta}=2X^T(X\theta-y)$
&emsp;&emsp;$\theta=(X^TX)^{-1}X^Ty$
There is no need to do feature scaling with the normal equation.
无需对正规方程进行特征缩放。
Gradient Descent | Normal Equation
---|---
Need to choose alpha|No need to choose alpha
Needs many iterations|No need to iterate
O($kn^2$) |O$(n^3)$,need to calculate inverse of $X^TX$ 
Works well when n is large	|Slow if n is very large

With the normal equation, computing the inversion has complexity $\mathcal{O}(n^3)$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.
使用正规方程，计算矩阵逆的复杂度为$\mathcal{O}(n^3)$。因此，如果我们具有大量特征，则正规方程将很慢。实际上，当n超过10,000时，可能是从正规方程转为迭代方式的好时机。
有人可能想说——**明明还可以继续化简啊！！！**
但实际的情况中，我们不能保证矩阵A总是方阵（square），但是$A^TA$总是可以保证是方阵。因为只有方阵才有逆矩阵，所以我们只能保证有$(A^TA)^{-1}$，而不能保证有$A^{-1}$。
然而，在现实生活中，$X^TX$往往不是满秩矩阵，例如在许多任务中我们会遇到大量的变量，其数目甚至超过样例数，导致$X$的列数多于行数，$X^TX$显然不满秩，此时可以解出多个$\theta$,选择哪一个解作为输出，将由学习算法的归纳偏好决定，常见的做法是引入**正则化(regularization)**.
*回忆一下，解线性方程组时，若因变量过多，则会解出多组解。*

---
### Normal Equation Noninvertibility
When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of $\theta$ even if $X^TX$ is not invertible. 
If $X^TX$ is **noninvertible**, the common causes might be having :

Redundant features, where two features are very closely related (i.e. they are linearly dependent)
Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" 
Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.