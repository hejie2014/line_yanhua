统计描述、数据可视化、概率模型、随机过程模拟

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785320709208-487c0db6-e8c6-4e10-860e-84ab03d18aa9.png)



# 1一元高斯分布Bk1_Ch36_01.py


```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

# 导入所需的库
import streamlit as st  # 用于创建交互式Web应用
from scipy.stats import beta  # 导入beta分布（虽然代码中未使用）
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # 用于数值计算

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

def uni_normal_pdf(x, mu, sigma):
    """
    计算一元正态分布（高斯分布）的概率密度函数（PDF）
    
    参数:
    x : array_like - 自变量值
    mu : float - 均值（位置参数）
    sigma : float - 标准差（尺度参数）
    
    返回:
    f_x : array_like - 对应的概率密度值
    """
    # 归一化系数：1/(sqrt(2*pi)*sigma)
    coeff = 1/np.sqrt(2*np.pi)/sigma
    # 标准化的z值：(x - mu)/sigma
    z = (x - mu)/sigma
    # 计算PDF：系数 * exp(-z^2/2)
    f_x = coeff*np.exp(-1/2*z**2)
    return f_x

# 生成x轴的数据点，范围从-5到5，共200个点
x_array = np.linspace(-5, 5, 200)

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Univariate Gaussian distribution PDF')  # 侧边栏标题
    # 显示高斯分布PDF的数学公式（LaTeX格式）
    st.latex(r'''{\displaystyle f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}
             e^{-{\frac {1}{2}}\left({
             \frac {x-\mu }{\sigma }}\right)^{2}}}''')
    
    # 创建滑块控件：调整均值mu，范围-5到5，步长0.2
    mu_input = st.slider('mu', min_value=-5.0, max_value=5.0, value=0.0, step=0.2)
    
    # 创建滑块控件：调整标准差sigma，范围0到4，步长0.1
    sigma_input = st.slider('sigma', min_value=0.0, max_value=4.0, value=1.0, step=0.1)

# 根据用户输入的参数计算高斯分布PDF值
pdf_array = uni_normal_pdf(x_array, mu_input, sigma_input)

# 创建图形和坐标轴，设置图形大小为8x5英寸
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制用户自定义参数的高斯分布PDF曲线（蓝色，线宽1）
ax.plot(x_array, pdf_array,
        'b', lw=1)

# 在均值位置添加红色虚线（表示均值位置）
ax.axvline(x=mu_input, c='r', ls='--')
# 在均值±标准差位置添加红色虚线（表示一个标准差范围）
ax.axvline(x=mu_input + sigma_input, c='r', ls='--')
ax.axvline(x=mu_input - sigma_input, c='r', ls='--')

# 绘制标准正态分布（均值=0，标准差=1）作为参考（浅灰色，线宽1）
ax.plot(x_array, uni_normal_pdf(x_array, 0, 1),
        c=[0.8, 0.8, 0.8], lw=1)

# 在标准正态分布的均值位置添加灰色虚线
ax.axvline(x=0, c=[0.8, 0.8, 0.8], ls='--')
# 在标准正态分布的均值±标准差位置添加灰色虚线
ax.axvline(x=0 + 1, c=[0.8, 0.8, 0.8], ls='--')
ax.axvline(x=0 - 1, c=[0.8, 0.8, 0.8], ls='--')

# 设置坐标轴范围和标签
ax.set_xlim(-5, 5)  # x轴范围：-5到5
ax.set_ylim(0, 1)   # y轴范围：0到1
ax.set_xlabel(r'$x$')  # x轴标签
ax.set_ylabel(r'$f_X(x)$')  # y轴标签（概率密度函数）

# 隐藏右侧和顶部的坐标轴线（使图形更简洁）
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

# 设置刻度位置（左侧和底部）
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# 设置刻度方向（向内）
ax.tick_params(axis="x", direction='in')
ax.tick_params(axis="y", direction='in')

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785320953313-eb847bb0-b73c-4853-8218-01d33eb55ae9.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785320961216-d4493325-ce00-49c7-ae1a-b7852e03ed34.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785320965603-a661beb4-b412-4379-8e1f-25e38c6f0e37.png)



# 2二元高斯分布Bk1_Ch36_02.py


```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

# 导入所需的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from scipy.stats import multivariate_normal  # 导入多元正态分布函数
import streamlit as st  # 用于创建交互式Web应用

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Bivariate Gaussian Distribution')  # 侧边栏标题：二元高斯分布
    
    # 创建滑块控件：调整第一个维度的均值 μ₁，范围-4到4，步长0.1
    mu_X1 = st.slider('mu_X1', min_value=-4.0, 
                      max_value=4.0, 
                      value=0.0, step=0.1)
    
    # 创建滑块控件：调整第二个维度的均值 μ₂，范围-4到4，步长0.1
    mu_X2 = st.slider('mu_X2', min_value=-4.0, 
                      max_value=4.0, 
                      value=0.0, step=0.1)
    
    # 创建滑块控件：调整第一个维度的标准差 σ₁，范围0.5到3，步长0.1
    sigma_X1 = st.slider('sigma_X1', min_value=0.5, 
                         max_value=3.0, 
                         value=1.0, step=0.1)
    
    # 创建滑块控件：调整第二个维度的标准差 σ₂，范围0.5到3，步长0.1
    sigma_X2 = st.slider('sigma_X2', min_value=0.5, 
                         max_value=3.0, 
                         value=1.0, step=0.1)
    
    # 创建滑块控件：调整相关系数 ρ，范围-0.95到0.95，步长0.05
    # 注意：范围限制在±0.95避免协方差矩阵奇异
    rho = st.slider('rho', min_value=-0.95, 
                    max_value=0.95, 
                    value=0.0, step=0.05)

# 定义均值向量（质心位置）
mu = [mu_X1, mu_X2]

# 构建协方差矩阵
# Σ = [[σ₁²,     ρ·σ₁·σ₂],
#      [ρ·σ₁·σ₂, σ₂²    ]]
Sigma = [[sigma_X1**2, sigma_X1*sigma_X2*rho], 
         [sigma_X1*sigma_X2*rho, sigma_X2**2]]

# 定义坐标范围：从 -width 到 width
width = 4
# 在x₁和x₂方向上生成等间距的网格点，每个方向321个点
x1 = np.linspace(-width, width, 321)
x2 = np.linspace(-width, width, 321)

# 创建二维网格坐标矩阵
# xx1: 每个点的x₁坐标，xx2: 每个点的x₂坐标
xx1, xx2 = np.meshgrid(x1, x2)

# 将xx1和xx2堆叠成三维数组，形状为 (321, 321, 2)
# 这样每个点都有(x₁, x₂)坐标对
xx12 = np.dstack((xx1, xx2))

# 创建多元正态分布对象（均值为mu，协方差为Sigma）
bi_norm = multivariate_normal(mu, Sigma)

# 计算每个网格点的二元高斯概率密度函数值
PDF_joint = bi_norm.pdf(xx12)

# 创建图形和坐标轴，设置图形大小为5x5英寸（正方形）
fig, ax = plt.subplots(figsize=(5, 5))

# 绘制二元高斯PDF的填充等高线图
# xx1, xx2: 网格坐标
# PDF_joint: 对应的高度值（概率密度）
# 20: 等高线层级数
# cmap='RdYlBu_r': 使用红-黄-蓝颜色映射（反转）
plt.contourf(xx1, xx2, PDF_joint, 20, cmap='RdYlBu_r')

# 在均值位置添加垂直虚线（表示μ₁的位置）
plt.axvline(x=mu_X1, color='k', linestyle='--')
# 在均值位置添加水平虚线（表示μ₂的位置）
plt.axhline(y=mu_X2, color='k', linestyle='--')

# 设置坐标轴标签
ax.set_xlabel('$x_1$')  # x轴标签
ax.set_ylabel('$x_2$')  # y轴标签

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321079412-980ff37f-d997-4e6f-89ad-6cf6bfa9f2e9.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321198181-f2652c15-cf29-403d-b31a-55ffc02e7b7c.png)

# 3三元高斯分布Bk1_Ch36_03.py
```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2023
###############

# 导入所需的库
import plotly.graph_objects as go  # Plotly的图形对象，用于创建交互式3D可视化
import numpy as np  # 用于数值计算
import streamlit as st  # 用于创建交互式Web应用
from scipy.stats import multivariate_normal  # 导入多元正态分布函数

# 在Streamlit中显示三元高斯分布的PDF公式（LaTeX格式）
st.latex(r'''{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})=
         {\frac {\exp \left(-{\frac {1}{2}}
         ({\mathbf {x} }-{\boldsymbol {\mu }})
         ^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}
         ({\mathbf {x} }-{\boldsymbol {\mu }})\right)}
         {\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}''')

def bmatrix(a):
    """
    将NumPy数组转换为LaTeX的bmatrix（矩阵）格式
    
    参数:
    a : numpy数组 - 要转换的矩阵
    
    返回:
    str - LaTeX格式的bmatrix字符串
    
    用途: 在Streamlit中美观地显示协方差矩阵
    """
    # 检查数组维度，最多支持2维
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    
    # 将数组转换为字符串并移除方括号，按行分割
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    
    # 构建LaTeX bmatrix
    rv = [r'\begin{bmatrix}']  # 矩阵开始标记
    # 每行元素用 & 连接，行末添加 \\
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']   # 矩阵结束标记
    return '\n'.join(rv)

# 使用np.mgrid生成三维网格坐标
# 范围从-3到3，步长0.2，生成三个维度的网格
# xxx1, xxx2, xxx3 的形状相同，每个维度30个点 (3-(-3))/0.2 = 30
xxx1, xxx2, xxx3 = np.mgrid[-3:3:0.2, -3:3:0.2, -3:3:0.2] 

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Trivariate Gaussian Distribution')  # 侧边栏标题：三元高斯分布
    
    # 创建滑块控件：调整三个维度的标准差
    # σ₁, σ₂, σ₃ 范围0.5到3，步长0.1
    sigma_1 = st.slider('sigma_1', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    sigma_2 = st.slider('sigma_2', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    sigma_3 = st.slider('sigma_3', min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    
    # 创建滑块控件：调整三个相关系数
    # ρ₁₂: x₁和x₂的相关系数
    # ρ₁₃: x₁和x₃的相关系数  
    # ρ₂₃: x₂和x₃的相关系数
    # 范围限制在±0.95避免协方差矩阵奇异
    rho_1_2 = st.slider('rho_1_2', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)
    rho_1_3 = st.slider('rho_1_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)
    rho_2_3 = st.slider('rho_2_3', min_value=-0.95, max_value=0.95, value=0.0, step=0.05)

# 构建协方差矩阵 (3x3对称矩阵)
# Σ = [[σ₁²,     ρ₁₂·σ₁·σ₂,  ρ₁₃·σ₁·σ₃],
#      [ρ₁₂·σ₁·σ₂, σ₂²,      ρ₂₃·σ₂·σ₃],
#      [ρ₁₃·σ₁·σ₃, ρ₂₃·σ₂·σ₃, σ₃²    ]]
SIGMA = np.array([[sigma_1**2, rho_1_2*sigma_1*sigma_2, rho_1_3*sigma_1*sigma_3],
                  [rho_1_2*sigma_1*sigma_2, sigma_2**2, rho_2_3*sigma_2*sigma_3],
                  [rho_1_3*sigma_1*sigma_3, rho_2_3*sigma_2*sigma_3, sigma_3**2]])

# 在Streamlit中显示协方差矩阵的LaTeX格式
st.latex(r'\Sigma = ' + bmatrix(SIGMA))

# 定义均值向量（这里固定为零向量）
# 注意：为了让可视化更集中，均值固定在原点
MU = np.array([0, 0, 0])

# st.write(xxx1.shape)  # 注释掉的调试代码，用于查看网格形状

# 将三维网格坐标重组为点集
# ravel()将多维数组展平为一维
# dstack沿第三维堆叠，形成 (N, 3) 的坐标数组
# 其中 N = 30 * 30 * 30 = 27000 个点
pos = np.dstack((xxx1.ravel(), xxx2.ravel(), xxx3.ravel()))

# st.write(pos.shape)  # 注释掉的调试代码，用于查看点集形状

# 创建多元正态分布对象（均值为MU，协方差为SIGMA）
rv = multivariate_normal(MU, SIGMA)

# 计算所有网格点的概率密度函数值
PDF = rv.pdf(pos)  # PDF是一维数组，长度27000

# 创建Plotly的3D体积可视化图形
fig = go.Figure(data=go.Volume(
    x=xxx1.flatten(),  # x坐标：展平网格
    y=xxx2.flatten(),  # y坐标：展平网格
    z=xxx3.flatten(),  # z坐标：展平网格
    value=PDF.flatten(),  # 每个点的PDF值（用于颜色映射）
    isomin=0,  # 体积显示的最小值（只显示PDF>0的区域）
    isomax=PDF.max(),  # 体积显示的最大值
    colorscale='RdYlBu_r',  # 颜色映射：红-黄-蓝（反转）
    opacity=0.1,  # 透明度（0.1使内部结构可见）
    surface_count=11,  # 等值面的数量（显示11个等密度面）
))

# 更新图形布局
fig.update_layout(
    scene=dict(
        xaxis_title=r'x_1',  # x轴标签
        yaxis_title=r'x_2',  # y轴标签
        zaxis_title=r'x_3',  # z轴标签
    ),
    width=1000,  # 图形宽度
    margin=dict(r=20, b=10, l=10, t=10)  # 图形边距
)

# 在Streamlit中显示Plotly交互式图表
# theme=None: 不使用Streamlit主题
# use_container_width=True: 使用容器宽度
st.plotly_chart(fig, theme=None, use_container_width=True)
```

# <!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321246491-ebaf87ef-2cb0-4042-b8ac-ba90a4d0ce12.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321258652-2a81e426-158a-4317-b3b4-9ec60ac7dddc.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321266815-404e70ac-1a71-41fa-b72e-47992cbbb242.png)
# 4 多项式回归Bk1_Ch36_04.py
```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# 导入所需的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.preprocessing import PolynomialFeatures  # 用于生成多项式特征
from sklearn.linear_model import LinearRegression  # 用于线性回归模型
import streamlit as st  # 用于创建交互式Web应用

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# 生成随机数据（带有噪声的正弦波）
np.random.seed(0)  # 设置随机种子以确保结果可复现
num = 30  # 数据点数量
X = np.random.uniform(0, 4, num)  # 在[0,4]区间生成30个均匀分布的随机数（特征）
y = np.sin(0.4*np.pi * X) + 0.4 * np.random.randn(num)  # 生成目标值：正弦波 + 噪声
data = np.column_stack([X, y])  # 将X和y合并为二维数组（用于后续绘图）

x_array = np.linspace(0, 4, 101).reshape(-1, 1)  # 生成0到4之间的101个点，用于绘制平滑曲线
degree_array = [1, 2, 3, 4, 7, 8]  # 预设的多项式次数（本代码中未实际使用）

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Polynomial Regression')  # 侧边栏标题：多项式回归
    # 创建滑块控件：调整多项式次数，范围1到9，默认值2，步长1
    degree = st.slider('Degree',
                       min_value=1, 
                       max_value=9, 
                       value=2, step=1)

# 创建图形和坐标轴，设置图形大小为5x5英寸（正方形）
fig, ax = plt.subplots(figsize=(5, 5))

# 创建多项式特征生成器，指定多项式的次数
poly = PolynomialFeatures(degree=degree)
# 将原始特征X转换为多项式特征矩阵 [1, x, x², x³, ..., x^degree]
X_poly = poly.fit_transform(X.reshape(-1, 1))

# 创建并训练线性回归模型（在多项式特征空间中进行线性回归）
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)  # 拟合多项式回归模型

# 预测训练集上的值
y_poly_pred = poly_reg.predict(X_poly)
data_ = np.column_stack([X, y_poly_pred])  # 合并预测值用于绘图

# 预测x_array（密集点）上的值，用于绘制平滑曲线
y_array_pred = poly_reg.predict(poly.fit_transform(x_array))

# === 绘图部分 ===

# 绘制原始数据点（散点图，蓝色圆圈，大小为20）
ax.scatter(X, y, s=20)

# 绘制预测值点（黑色叉号）
ax.scatter(X, y_poly_pred, marker='x', color='k')

# 绘制从原始点到预测点的连线（显示拟合误差）
# 使用列表推导式提取data_和data的x坐标和y坐标
ax.plot(([i for (i,j) in data_], [i for (i,j) in data]),
        ([j for (i,j) in data_], [j for (i,j) in data]),
        c=[0.6,0.6,0.6], alpha=0.5)  # 灰色半透明连线

# 绘制拟合曲线（红色平滑曲线）
ax.plot(x_array, y_array_pred, color='r')

# === 提取和显示回归方程 ===

# 提取模型参数
coef = poly_reg.coef_  # 系数数组 [θ₁, θ₂, ..., θ_degree]
intercept = poly_reg.intercept_  # 截距项 θ₀

# 构建回归方程的字符串表示
equation = '$y = {:.1f}'.format(intercept)  # 从截距开始
for j in range(1, len(coef)):  # 遍历每个系数
    equation += ' + {:.1f}x^{}'.format(coef[j], j)  # 添加项: θ_j * x^j
equation += '$'  # 闭合LaTeX数学模式

# 将 "+ -" 替换为 "-" 使方程更美观（处理负系数）
equation = equation.replace("+ -", "-")

# 在图表上显示回归方程（注释掉，因为使用了st.write在Web界面显示）
# ax.text(0.05, -1.8, equation)

# 使用Streamlit在Web界面上显示回归方程（支持LaTeX渲染）
st.write(equation)

# === 设置图表格式 ===

ax.set_aspect('equal', adjustable='box')  # 设置等比例坐标轴
ax.set_xlim(0, 4)  # x轴范围0到4
ax.grid(False)  # 关闭网格线（让图表更清晰）
ax.set_ylim(-2, 2)  # y轴范围-2到2

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321368256-3a362b9c-5b7b-4f9e-bf3b-d08d72e13570.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321415572-e40c79c7-16c8-4bbf-a99d-d5c148093cb4.png)



# 5 主成分分析Bk1_Ch36_05.py


```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# 导入所需的库
import pandas as pd  # 用于数据处理和DataFrame操作
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# 导入其他所需库
import pandas_datareader as pdr  # 用于从FRED等数据源读取经济数据
# pip install pandas_datareader  # 安装命令（注释）
import seaborn as sns  # 用于数据可视化（基于matplotlib）
import statsmodels.multivariate.pca as pca  # statsmodels中的PCA实现
import streamlit as st  # 用于创建交互式Web应用

# 重新设置绘图参数（覆盖之前的设置，确保一致性）
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

# 从FRED（美联储经济数据）读取国债收益率数据
# 数据系列说明：
# DGS6MO: 6个月期国债收益率
# DGS1:   1年期国债收益率
# DGS2:   2年期国债收益率
# DGS5:   5年期国债收益率
# DGS7:   7年期国债收益率
# DGS10:  10年期国债收益率
# DGS20:  20年期国债收益率
# DGS30:  30年期国债收益率
df = pdr.data.DataReader(['DGS6MO','DGS1',
                          'DGS2','DGS5',
                          'DGS7','DGS10',
                          'DGS20','DGS30'], 
                          data_source='fred',  # 数据源：FRED
                          start='01-01-2022',  # 开始日期
                          end='12-31-2022')    # 结束日期

df = df.dropna()  # 删除缺失值（NaN）

# 重命名列，使用更直观的标签（期限年份）
df = df.rename(columns={'DGS6MO': '0.5 yr', 
                        'DGS1': '1 yr',
                        'DGS2': '2 yr',
                        'DGS5': '5 yr',
                        'DGS7': '7 yr',
                        'DGS10': '10 yr',
                        'DGS20': '20 yr',
                        'DGS30': '30 yr'})

# 计算每日收益率变化（百分比变化）
# 将收益率水平值转换为日变化量（差分）
X_df = df.pct_change()  # 计算百分比变化
X_df = X_df.dropna()    # 删除第一个NaN值（因pct_change产生）

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Principal Component Analysis')  # 侧边栏标题：主成分分析
    # 创建滑块控件：选择要保留的主成分数量，范围1-8，默认值2
    num_of_PCs = st.slider('Number of PCs',
                           min_value=1, 
                           max_value=8, 
                           value=2, step=1)

# 创建PCA模型对象，对数据进行标准化处理（standardize=True）
# 标准化使每个特征具有均值为0、方差为1，避免量纲影响
pca_model = pca.PCA(X_df, standardize=True)

# 获取特征值（主成分的方差）
variance_V = pca_model.eigenvals 

# 计算每个主成分的方差解释比例
# 公式：各主成分方差 / 总方差
explained_var_ratio = pca_model.eigenvals / pca_model.eigenvals.sum()

# 将原始数据投影到选定的主成分空间
# 得到降维后的数据（保留num_of_PCs个主成分）
X_df_ = pca_model.project(num_of_PCs)

# 创建2行4列的子图网格，总尺寸8x4英寸
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()  # 将2x4的axes数组展平为8个一维数组

# 遍历每个变量（8个不同期限的国债收益率）
for col_idx, ax_idx in zip(list(X_df_.columns), axes):
    # 绘制主成分重构后的数据（降维后的近似）
    sns.lineplot(X_df_[col_idx], ax=ax_idx)
    
    # 绘制原始数据（实际观测值）
    sns.lineplot(X_df[col_idx], ax=ax_idx)
    
    # 绘制残差（原始数据 - 重构数据）
    # 黑色线条显示PCA未能捕捉的信息
    sns.lineplot(X_df[col_idx] - X_df_[col_idx], c='k', ax=ax_idx)
    
    # 隐藏x轴和y轴刻度（使图表更简洁）
    ax_idx.set_xticks([])
    ax_idx.set_yticks([])
    
    # 在y=0处添加水平参考线（帮助观察残差的分布）
    ax_idx.axhline(y=0, c='k')

# 注释掉的代码（原计划绘制散点图，但未使用）
# ax_idx.plot([-0.3, 0.3],[-0.3, 0.3],c = 'r')
# ax_idx.set_aspect('equal', adjustable='box')
# ax_idx.set_xlim(-0.3, 0.3)
# ax_idx.set_ylim(-0.3, 0.3)

plt.tight_layout()  # 自动调整子图间距，避免重叠

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321491295-6f33bf7a-7860-4da6-9f1c-84b569c84bf2.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321538958-f50726da-0ced-4544-be6f-46ce36256e72.png)

# 6最近邻分类k1_Ch36_06.py
```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# 导入所需的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于数据可视化（基于matplotlib，提供更美观的样式）
from matplotlib.colors import ListedColormap  # 用于创建自定义颜色映射
from sklearn import neighbors, datasets  # sklearn的KNN分类器和鸢尾花数据集
import streamlit as st  # 用于创建交互式Web应用

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('k-Nearest Neighbors Classification')  # 侧边栏标题：k近邻分类
    # 创建滑块控件：调整k近邻数量（k值），范围2-20，默认值5
    k_neighbors = st.slider('k_neighbors', 
                            min_value=2, 
                            max_value=20, 
                            value=5, step=1)

# === 加载和准备数据 ===
# 加载鸢尾花（Iris）数据集
iris = datasets.load_iris()
# 只使用前两个特征：萼片长度（sepal length）和萼片宽度（sepal width）
X = iris.data[:, :2]  # 特征矩阵
y = iris.target       # 目标标签（0: setosa, 1: versicolor, 2: virginica）

# === 生成用于决策边界可视化的网格数据 ===
# 在特征空间创建密集的网格点
x1_array = np.linspace(4, 8, 101)  # 萼片长度范围：4-8cm
x2_array = np.linspace(1, 5, 101)  # 萼片宽度范围：1-5cm
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # 创建二维网格坐标

# === 自定义颜色映射 ===
# 创建浅色颜色映射（用于填充决策区域）
rgb = [[255, 238, 255],   # 浅粉色 - setosa区域
       [219, 238, 244],   # 浅蓝色 - versicolor区域
       [228, 228, 228]]   # 浅灰色 - virginica区域
rgb = np.array(rgb)/255.  # 归一化到[0,1]范围
cmap_light = ListedColormap(rgb)  # 创建浅色颜色映射

# 创建深色颜色映射（用于数据点）
cmap_bold = [[255, 51, 0],     # 红色 - setosa
             [0, 153, 255],    # 蓝色 - versicolor
             [138, 138, 138]]  # 灰色 - virginica
cmap_bold = np.array(cmap_bold)/255.  # 归一化到[0,1]范围

# === 创建和训练kNN分类器 ===
# 创建k近邻分类器对象，指定近邻数量k
kNN = neighbors.KNeighborsClassifier(k_neighbors)
kNN.fit(X, y)  # 使用训练数据拟合模型

# 将网格点重组为查询点集（形状：10201 x 2）
q = np.c_[xx1.ravel(), xx2.ravel()]

# 对网格上的每个点进行预测（获得分类标签）
y_predict = kNN.predict(q)
# 将预测结果重塑为网格形状（101 x 101），用于绘制决策边界
y_predict = y_predict.reshape(xx1.shape)

# === 可视化决策边界和数据点 ===
fig, ax = plt.subplots()  # 创建图形和坐标轴

# 绘制填充的决策区域（使用浅色颜色映射）
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)

# 绘制决策边界线（在类别边界处绘制等高线）
# levels=[0,1,2] 指定在类别边界绘制
plt.contour(xx1, xx2, y_predict, levels=[0, 1, 2], 
            colors=np.array([0, 68, 138])/255.)  # 深蓝色边界线

# 绘制原始数据点（散点图）
sns.scatterplot(x=X[:, 0], y=X[:, 1],  # x和y坐标
                hue=iris.target_names[y],  # 根据类别着色
                ax=ax,
                palette=dict(setosa=cmap_bold[0,:],      # 红色点
                            versicolor=cmap_bold[1,:],    # 蓝色点
                            virginica=cmap_bold[2,:]),    # 灰色点
                alpha=1.0,  # 不透明
                linewidth=1,  # 点边框宽度
                edgecolor=[1, 1, 1])  # 白色边框，使点更清晰

# === 设置图表格式 ===
plt.xlim(4, 8)  # x轴范围：4-8
plt.ylim(1, 5)  # y轴范围：1-5
plt.xlabel(iris.feature_names[0])  # x轴标签：萼片长度
plt.ylabel(iris.feature_names[1])  # y轴标签：萼片宽度

# 设置网格样式（虚线，宽度0.25，灰色）
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])

# 设置等比例坐标轴（x和y单位长度相同）
ax.set_aspect('equal', adjustable='box')

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

# <!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321575596-ef59efcc-5bde-4557-a0d7-d79c9302e002.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321582933-35767588-19ae-4f77-b945-8ebd7ad5b438.png)
# 7支持向量机 + 高斯核Bk1_Ch36_07.py


```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# 导入所需的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于数据可视化（基于matplotlib，提供更美观的样式）
from matplotlib.colors import ListedColormap  # 用于创建自定义颜色映射
from sklearn import neighbors, datasets  # 导入sklearn的邻近算法和数据集的模块
import streamlit as st  # 用于创建交互式Web应用

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# Streamlit侧边栏 - 用于用户交互控制
with st.sidebar:
    st.title('Support Vector Machine')  # 侧边栏标题：支持向量机
    # 高斯核参数Gamma
    st.write('Gaussian kernel')  # 显示文本：高斯核
    # 创建滑块控件：调整高斯核的gamma参数，范围0.001到5.0，默认值1.0
    gamma = st.slider('Gamma', 
                       min_value=0.001, 
                       max_value=5.0, 
                       value=1.0, step=0.05)

# === 加载和准备数据 ===
# 加载鸢尾花（Iris）数据集
iris = datasets.load_iris()
# 只使用前两个特征：萼片长度（sepal length）和萼片宽度（sepal width）
X = iris.data[:, :2]  # 特征矩阵
y = iris.target       # 目标标签（0: setosa, 1: versicolor, 2: virginica）

# === 生成用于决策边界可视化的网格数据 ===
# 在特征空间创建密集的网格点
x1_array = np.linspace(4, 8, 101)  # 萼片长度范围：4-8cm
x2_array = np.linspace(1, 5, 101)  # 萼片宽度范围：1-5cm
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # 创建二维网格坐标

# === 自定义颜色映射 ===
# 创建浅色颜色映射（用于填充决策区域）
rgb = [[255, 238, 255],   # 浅粉色 - setosa区域
       [219, 238, 244],   # 浅蓝色 - versicolor区域
       [228, 228, 228]]   # 浅灰色 - virginica区域
rgb = np.array(rgb)/255.  # 归一化到[0,1]范围
cmap_light = ListedColormap(rgb)  # 创建浅色颜色映射

# 创建深色颜色映射（用于数据点）
cmap_bold = [[255, 51, 0],     # 红色 - setosa
             [0, 153, 255],    # 蓝色 - versicolor
             [138, 138, 138]]  # 灰色 - virginica
cmap_bold = np.array(cmap_bold)/255.  # 归一化到[0,1]范围

# 将网格点重组为查询点集（形状：10201 x 2）
q = np.c_[xx1.ravel(), xx2.ravel()]

# === 导入SVM模块 ===
from sklearn import svm

# 创建支持向量机分类器对象（使用高斯核/RBF核）
# kernel='rbf': 使用径向基函数（高斯）核
# gamma: 核函数的宽度参数，控制单个训练样本的影响范围
SVM = svm.SVC(kernel='rbf', gamma=gamma)

# 用训练数据训练SVM模型
SVM.fit(X, y)

# 用训练好的SVM对网格上的每个点进行预测（获得分类标签）
y_predict = SVM.predict(q)
# 将预测结果重塑为网格形状（101 x 101），用于绘制决策边界
y_predict = y_predict.reshape(xx1.shape)

# === 可视化决策边界和数据点 ===
fig, ax = plt.subplots()  # 创建图形和坐标轴

# 绘制填充的决策区域（使用浅色颜色映射）
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)

# 绘制决策边界线（在类别边界处绘制等高线）
# levels=[0,1,2] 指定在类别边界绘制
plt.contour(xx1, xx2, y_predict, levels=[0, 1, 2], 
            colors=np.array([0, 68, 138])/255.)  # 深蓝色边界线

# 绘制原始数据点（散点图）
sns.scatterplot(x=X[:, 0], y=X[:, 1],  # x和y坐标
                hue=iris.target_names[y],  # 根据类别着色
                ax=ax,
                palette=dict(setosa=cmap_bold[0,:],      # 红色点
                            versicolor=cmap_bold[1,:],    # 蓝色点
                            virginica=cmap_bold[2,:]),    # 灰色点
                alpha=1.0,  # 不透明
                linewidth=1,  # 点边框宽度
                edgecolor=[1, 1, 1])  # 白色边框，使点更清晰

# === 设置图表格式 ===
plt.xlim(4, 8)  # x轴范围：4-8
plt.ylim(1, 5)  # y轴范围：1-5
plt.xlabel(iris.feature_names[0])  # x轴标签：萼片长度
plt.ylabel(iris.feature_names[1])  # y轴标签：萼片宽度

# 设置网格样式（虚线，宽度0.25，灰色）
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])

# 设置等比例坐标轴（x和y单位长度相同）
ax.set_aspect('equal', adjustable='box')

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321663478-fa629881-946f-4466-bb41-01661dd70bc0.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321716228-7af463c0-ed9f-4b76-a6b5-f46c258914d3.png)

# 8高斯混合模型聚类Bk1_Ch36_08.py


```python
###############
# Authored by Weisheng Jiang
# Book 1  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# 导入所需的库
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.colors import ListedColormap  # 用于创建自定义颜色映射
import numpy as np  # 用于数值计算
from sklearn import datasets  # 用于加载鸢尾花数据集
from sklearn.mixture import GaussianMixture  # 高斯混合模型（GMM）
import streamlit as st  # 用于创建交互式Web应用
from matplotlib.patches import Ellipse  # 用于绘制椭圆

# 设置matplotlib的全局绘图参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]  # 设置字体为Roboto
p["font.weight"] = "light"  # 字体粗细设置为light
p["ytick.minor.visible"] = True  # 显示y轴次要刻度
p["xtick.minor.visible"] = True  # 显示x轴次要刻度
p["axes.grid"] = True  # 显示网格
p["grid.color"] = "0.5"  # 网格颜色为灰色（0.5灰度）
p["grid.linewidth"] = 0.5  # 网格线宽度

# ============================================================================
# 辅助函数：绘制高斯混合模型的椭圆、主轴和中心点
# ============================================================================
def make_ellipses(gmm, ax):
    """
    绘制GMM每个高斯分量的置信椭圆、主成分轴和中心点
    
    参数:
    gmm : GaussianMixture - 训练好的GMM模型
    ax : matplotlib.axes.Axes - 绘图坐标轴
    """
    # 遍历每个高斯分量（簇）
    for j in range(0, K):
        # 根据协方差类型获取协方差矩阵
        if gmm.covariance_type == 'full':
            # 完整协方差矩阵：每个分量有自己的完整协方差矩阵
            covariances = gmm.covariances_[j]
        elif gmm.covariance_type == 'tied':
            # 绑定协方差：所有分量共享同一个协方差矩阵
            covariances = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            # 对角协方差：每个分量有对角协方差矩阵（特征独立）
            covariances = np.diag(gmm.covariances_[j])
        elif gmm.covariance_type == 'spherical':
            # 球形协方差：每个分量有各向同性的方差
            covariances = np.eye(gmm.means_.shape[1]) 
            covariances = covariances * gmm.covariances_[j]
        
        # 使用奇异值分解（SVD）进行特征值分解
        # U: 特征向量矩阵, S: 奇异值（特征值）, V_T: 转置的特征向量
        U, S, V_T = np.linalg.svd(covariances)
        
        # 计算椭圆的长轴和短轴长度（2倍标准差）
        # 注意：在GMM中，协方差矩阵的特征值平方根表示标准差
        major, minor = 2 * np.sqrt(S)  # 2倍标准差覆盖约95%的数据
        
        # 计算椭圆长轴的旋转角度（弧度转角度）
        angle = np.arctan2(U[1, 0], U[0, 0])  # 第一个主成分的方向
        angle = 180 * angle / np.pi  # 转换为度
        
        # 绘制高斯分量的中心点（黑色叉号）
        ax.plot(gmm.means_[j, 0], gmm.means_[j, 1],
                color='k', marker='x', markersize=10)

        # 绘制半长轴向量（第一个主成分方向）
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1],
                  U[0, 0], U[1, 0], scale=5/major)  # 缩放以适配图形

        # 绘制半短轴向量（第二个主成分方向）
        ax.quiver(gmm.means_[j, 0], gmm.means_[j, 1], 
                  U[0, 1], U[1, 1], scale=5/minor)  # 缩放以适配图形
        
        # 绘制多个置信椭圆（1σ, 2σ, 3σ）
        # scale=1: 1倍标准差（约68%数据）
        # scale=2: 2倍标准差（约95%数据）
        # scale=3: 3倍标准差（约99.7%数据）
        for scale in np.array([3, 2, 1]):
            ell = Ellipse(gmm.means_[j, :2], 
                          scale * major,  # 椭圆宽度（长轴）
                          scale * minor,  # 椭圆高度（短轴）
                          angle,          # 旋转角度
                          color=rgb[j, :],  # 使用预定义颜色
                          alpha=0.18)      # 透明度
            ax.add_artist(ell)  # 将椭圆添加到图中


# ============================================================================
# 数据准备
# ============================================================================

# 生成用于决策边界可视化的网格数据
x1_array = np.linspace(4, 8, 101)  # 萼片长度范围：4-8cm
x2_array = np.linspace(1, 5, 101)  # 萼片宽度范围：1-5cm
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # 创建二维网格坐标

# 加载鸢尾花数据集，只使用前两个特征（萼片长度和宽度）
iris = datasets.load_iris()
X = iris.data[:, :2]  # 特征矩阵

# 定义四种协方差类型
covariance_types = ['tied', 'spherical', 'diag', 'full']

# ============================================================================
# Streamlit侧边栏 - 用户交互控制
# ============================================================================
with st.sidebar:
    st.title('GMM Clustering')  # 侧边栏标题：高斯混合模型聚类
    # 创建单选按钮：选择协方差类型
    covariance_type = st.radio('covariance_type', covariance_types)

# 簇的数量（固定为3，对应鸢尾花的3个种类）
K = 3

# ============================================================================
# 自定义颜色映射
# ============================================================================
rgb = [[255, 51, 0],      # 红色 - 第一个簇
       [0, 153, 255],    # 蓝色 - 第二个簇
       [138, 138, 138]]  # 灰色 - 第三个簇
rgb = np.array(rgb) / 255.  # 归一化到[0,1]范围
cmap_bold = ListedColormap(rgb)  # 创建颜色映射

# ============================================================================
# 训练高斯混合模型
# ============================================================================

# 创建GMM对象
# n_components: 高斯分量数量（3个簇）
# covariance_type: 协方差矩阵类型（用户选择）
gmm = GaussianMixture(n_components=K, covariance_type=covariance_type)

# 拟合数据
gmm.fit(X)

# 对网格上的每个点进行预测（获得簇标签）
Z = gmm.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)  # 重塑为网格形状

# ============================================================================
# 可视化
# ============================================================================

# 创建图形，尺寸10x5英寸（左侧：椭圆可视化，右侧：决策边界）
fig = plt.figure(figsize=(10, 5))

# ----- 左侧子图：显示GMM的椭圆和主成分方向 -----
ax = fig.add_subplot(1, 2, 1)

# 绘制原始数据点（深蓝色，白色边框）
ax.scatter(x=X[:, 0], y=X[:, 1], 
           color=np.array([0, 68, 138])/255., 
           alpha=1.0,
           linewidth=1, edgecolor=[1, 1, 1])

# 调用函数绘制椭圆、主轴和中心点
make_ellipses(gmm, ax)

# 设置坐标轴范围和标签
ax.set_xlim(4, 8)
ax.set_ylim(1, 5)
ax.set_xlabel(iris.feature_names[0])  # 萼片长度
ax.set_ylabel(iris.feature_names[1])  # 萼片宽度

# 设置网格样式
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
ax.set_aspect('equal', adjustable='box')  # 等比例坐标轴

# ----- 右侧子图：显示聚类决策边界 -----
ax = fig.add_subplot(1, 2, 2)

# 绘制填充的决策区域（使用深色颜色映射，透明度0.18）
ax.contourf(xx1, xx2, Z, cmap=cmap_bold, alpha=0.18)

# 绘制决策边界线（在类别边界处绘制等高线）
ax.contour(xx1, xx2, Z, levels=[0, 1, 2], 
           colors=[np.array([0, 68, 138])/255.])

# 绘制原始数据点（深蓝色，白色边框）
ax.scatter(x=X[:, 0], y=X[:, 1], 
           color=np.array([0, 68, 138])/255., 
           alpha=1.0,
           linewidth=1, edgecolor=[1, 1, 1])

# 绘制每个簇的中心点（黑色叉号，大尺寸）
centroids = gmm.means_
ax.scatter(centroids[:, 0], centroids[:, 1], 
           marker="x", s=100, linewidths=1.5,
           color="k")

# 设置坐标轴范围和标签
ax.set_xlim(4, 8)
ax.set_ylim(1, 5)
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])

# 设置网格样式
ax.grid(linestyle='--', linewidth=0.25, color=[0.5, 0.5, 0.5])
ax.set_aspect('equal', adjustable='box')

# 使用Streamlit显示matplotlib图形
st.pyplot(fig)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321815675-169822e9-6e46-4d8e-9cb1-571d2c8f715a.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/51615940/1785321862220-b578ad24-ef67-4c82-a671-213a5c252d48.png)

