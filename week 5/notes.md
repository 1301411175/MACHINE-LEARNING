# 第四周
 
 ---

 ### Cost Function
 Let's first define a few variables that we will need to use:
 
 -  $L$ = total number of layers in the network
 -  $s_l$ = number of units (not counting bias unit) in layer l
 -  $K$ = number of output units/classes

We denote $h_\Theta(x)_k$ as being a hypothesis that results in the $k^{th}$ output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the **cost function for regularized logistic regression** was:
$$
J(\theta)=-\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$
For **neural networks**, it is going to be slightly more complicated:

$$
J(\Theta)=-\frac{1}{m}\sum^m_{i=1}\sum^K_{k=1}[y^{(i)}_klog((h_\Theta(x^{(i)}))_k) + (1-y^{(i)}_k)log(1-(h_
\Theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum^{(L-1)}_{(l=1)}\sum^{s_l}_{i=1}\sum*{s_l+1}_{j=1}(\Theta^{(l)}_{j,i})^2
$$

---

### Backpropagation Algorithm
"backpropagation" is netural-network terminology for minimizing cost function, just like what we were doing with gradient in logistic and linear regression. Our goal is compute: $min_\Theta J(\Theta)$.
That is ,we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equation we use to compute the partial derivative of $J(\Theta)$
$\frac{\partial}{\partial \Theta^{(i)}_{i,j}}J(\Theta)$.
To do this, we use the following algorithm;

**Back propagation Algorithm**
Given training set $\{(x^{(1)},y^{(1)}), \dots ,(x^{(m)},y^{(m)})\}$

- Set $\Delta^{(l)}_{(i,j)} := 0$ for all $(l,i,j)$ , (hence you end up having a matrix full of zeros)
  
For training example t=1 to m:
1. Set $a^{i}:=x^{(i)}$
2. Perform forward propagation to compute $a^{(l)}$ for $l=2,3,\dots,L$
![compute $a^{(l)}$](https://img.imgdb.cn/item/60588dce8322e6675c9f1318.jpg)
3. Using $y^{(t)}$, compute $\delta^{(L)}=a^{(L)}-y^{(t)}$.
    where L is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:
4. Compute $\delta^{(L-1)},\delta^{(L-2)},\dots, \delta^{(2)}$ using $\delta^{(l)}=((\Theta^{(l)})^T\delta^{(l+1)}).*a^{(l)}.*(1-a^{(l)})$
The dalta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$. We then element-wise multiply that with a function called $g'$, or $g-prime$, which is the derivative of the activation function $g$ evaluatded(评估;评价) with input values given by $z^{(l)}$.
The $g-prime$ derivative terms can also be written out as:
> **$g'(z) = a^{(l)} .* (1-a^{(l)})$**

5. $\Delta^{(l)}_{i,j}:=\Delta^{(l)}_{i,j}+a^{(l)}_j\delta^{(l+1)}_i$ or vectorization:
    $\Delta^{(l)}:=\Delta^{(l)}+\delta^(l+1)(a^{(l)})^T$

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac{\partial}{\partial \Theta^{(i)}_{i,j}}J(\Theta)=D^{(i)}_{i,j}$

---


   




