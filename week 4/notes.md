# 第四周
 
 ---

 ### Model Representation —— 模型表示
Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs **(dendrites)(树突)** as electrical inputs (called "spikes"(脉冲)) that are channeled(传送) to outputs **(axons)（轴突）**. In our model, our dendrites are like the input features $x_1\cdots x_n$, and the output is the result of our hypothesis function. In this model our $x_0$ input node is sometimes called the "bias unit(偏置单元)." It is always equal to 1. In neural networks, we use the same logistic function as in classification, $\frac{1}{1 + e^{-\theta^Tx}}$ , yet we sometimes call it a sigmoid (logistic) **activation** function. In this situation, our **"theta" parameters** are sometimes called **"weights"**.

Our input nodes (layer 1), also known as the **"input layer"(输入层)**, go into another node (layer 2), which finally outputs the hypothesis function, known as the **"output layer"(输出层)**.
We can have intermediate layers of nodes between the input and output layers called the **"hidden layers."(隐藏层)**

In this example, we label these intermediate or "hidden" layer nodes $a^2_0 \cdots a^2_n$ and call them **"activation units."(激活单元)**

> $a^{(j)}_i=$"activation" of unit i in layer j
> $\Theta(j)$=matrix of weights controlling function mapping from layer j to layer j+1

![image](https://img.imgdb.cn/item/604a36375aedab222c4d3675.jpg)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $\Theta^{(1)}$ is going to be 4×3 where $s_j = 2$ and $s_{j+1} = 4$, so $s_{j+1} \times (s_j + 1) = 4 \times 3$.

We're going to define a new variable $z_k^{(j)}$ that encompasses(包含) the parameters inside our $g$ function. In our previous example if we replaced by the variable z for all the parameters we would get:
> $a^{(2)}_1=g(z^{(2)}_1)$
> $a^{(2)}_2=g(z^{(2)}_2)$
> $a^{(2)}_3=g(z^{(2)}_3)$

In other words, for layer j=2 and node k, the variable z will be:
> $z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n$

The vector representation of x and $z^{j}$ is:
> $x=\left[\begin{matrix} x_0 \\ x_1 \\ \cdots \\ x_n \end{matrix}\right]$  
> $z^{(j)}=\left[\begin{matrix}   z_1^{(j)} \\ z_2^{(j)} \\ \cdots \\ z_n^{(j)} \end{matrix}\right]$

Setting $x = a^{(1)}$, we can rewrite the equation as:
> $z^{(j)}=\Theta^{(j−1)}a^{(j−1)}$

We are multiplying our matrix $\Theta^{(j-1)}$ with dimensions $s_j\times (n+1)$(where $s_j$ is the number of our activation nodes) by our vector $a^{(j-1)}$ with height (n+1). This gives us our vector $z^{(j)}$ with height $s_j$. Now we can get a vector of our activation nodes for layer j as follows:
> $a^{(j)}=g(z^{(j)})$

We can then add a bias unit (equal to 1) to layer j after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:
> $z^{(j+1)}=\Theta^{(j)}a^{(j)}$

We get this final z vector by multiplying the next theta matrix after $\Theta^{(j-1)}$ with the values of all the activation nodes we just got. This last theta matrix $\Theta^{(j)}$ will have **only one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. We then get our final result with:
> $h_Θ(x)=a^{(j+1)}=g(z^{(j+1)})$

Notice that in this **last step**, between layer j and layer j+1, we are doing exactly the **same thing** as we did in **logistic regression**. Adding all these intermediate(中间的) layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

---

### 