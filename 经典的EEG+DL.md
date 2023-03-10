## 写在前面的话

**:smile:**这是一篇**非常经典的利用深度学习网络来进行脑电信号解码研究**的论文，在Github网站搜索关键词"EEG",第一页便会出现这篇文章的复现工作。之所以在这里提出：发表我个人的理解，对其中的实验过程与代码进行解释，也欢迎能够看到的伙伴们一起来进行交流。

**:smile:**论文提出了一种新的网络结构与思考方式来处理脑电信号，要知道的是这是2016年发表ICLR上的文章，近些年来**ICLR也被誉为人工智能领域的TOP2的期刊**，文章将深度学习与脑电信号的结合从 **时域、频域、空间电极位置***，三个层面对脑电信号进行解剖，不断学习脑电信号中隐含的信息并利用深度学习网络进行学习，取得了非常好的解码效果。我个人认为这是这篇论文能发表在ICLR上的最重要原因！

**:smile:**从另一个角度来讲，这篇文章所提到的实验可以作为EEG-Decoding文档后的一个补充和论证，因为这完全是一个从**认知实验设计->数据采集->深度学习网络**的全过程。关于实验的详细过程放在后文介绍，文章的代码的实现过程具有pytorch和tf两个版本，但是pytorch的版本存在着一些问题，我也将在后面进行解释。

## 论文剖析

### 实验过程

:wink:工作记忆负责暂时保留信息，这对于大脑中的任何信息操作都至关重要。它的能力限制了个人在一系列认知功能中的能力。超过个人能力的认知需求（负荷）的增加导致超负荷状态，导致混乱和学习能力下降（Sweller等人，1998年）。因此，识别个人认知负荷的能力对于包括脑-机接口、人机交互和辅导服务在内的许多应用变得重要。

:wink:文章所做的实验是一个认知过程，这里使用了一个在工作记忆实验中获得的EEG数据集。15名参与者（8名女性）进行了标准的工作记忆实验，记录了脑电图。简而言之，在标准的10-10个位置放置64个电极，以500 Hz的采样频率记录连续EEG。电极沿内侧外侧轮廓以10%的距离放置。其中**两名受试者的数据被排除在数据集之外，因为他们记录的数据中存在过多的噪音和伪影**。在实验过程中，一组英文字符**(注意：是一组字符，里面可能包含2,4,6,8个字母)**被显示0.5秒（SET），参与者被要求记住这些字符。三秒钟后显示了一个TEST字符，参与者通过按下按钮指示测试字符是否在第一个数组中（“SET”）。每个参与者重复实验240次。每次试验的SET中的字符数随机选择为2、4、6或8。SET中字符数决定了参与者的认知负荷量，因为随着字符数的增加，需要更多的心理资源来保留信息。在整篇文章中，分别确定了包含2个、4个、6个和8个字符的每个条件，负载为1-4。在个人将信息保留在记忆中期间（3.5秒）记录的大脑活动被用来识别心理工作量的大小。下图展示了工作记忆实验的时间进程。分类任务是从脑电图记录中识别与设定大小（呈现给受试者的字符数）相对应的负荷水平。定义了与负荷1-4相对应的四个不同类别，并将从13名受试者采集的2670份样本分配给这四个类别。

![image-20230203151136393](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding-Cognition/images/image-20230203151136393.png)

### 数据结构与预处理

:sunglasses:15名受试者参与了实验，其中两名受试者的数据被剔除，可能是受到比较大的噪声影响，每个人重复240个试次，因此最终得到的**实验总数为13 x 240=3120个**，但是这其中又有很多是并没有回答正确的，因此再剔除错误回答后一共剩下2670个试次，并且在"FeatureMat_timeWin.mat"数据文件中最终储存的**格式为2670 x 1345，其中2670 x 1344是脑电信号，而2670 x 1是标签文件，至于为什么是1344，代码中给出了答案：1344=192 x 7，7代表将单次实验3.5s(**上面提到3.5s是记忆的加工过程**)的分为了7个窗口，每个窗口为0.5s，192=64*3,64代表的是64个电极位置，3代表的是α，β，θ频段的3个值，也就是说64个电极 每个电极位置有三个值。因此总结而言数据的结构为：7个窗口，每个窗口有192个数据，分别对应这一个窗口内64通道的3个频段的值**。

:question:但是令我不明白的是，这些数据应该是功率值，但是为什么会出现负数呢？如果不是功率值是电压值的话，为什么才1344个呢？

#### 时域特征

:+1:单次实验过程为：刺激显示0.5s，之后是3s的记忆时间，然后进行test字符检验，再进入下一次实验

#### 空间电极特征

:+1:空间特征即为eeg采样的空间电极位置，EEG电极在三维空间中分布在头皮上。为了将空间分布的活动图转换为二维图像，我们需要首先将电极的位置从三维空间投影到二维表面上。然而，这种变换还应保持相邻电极之间的相对距离。为此，文章中使用了方位角等距投影（AEP），也称为极投影，借用自地图应用（Snyder，1987）。该方法的缺点是，地图上的点之间的距离仅相对于单个点（中心点）保持，因此所有电极对之间的相对距离将不会精确保持。将AEP应用于三维电极位置，我们获得了电极的二维投影位置（图1）。图像的宽度和高度表示皮层上活动的空间分布。

![image-20230203153327536](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding-Cognition/images/image-20230203153327536.png)

:point_right:按照上述提到的方法，最终得到的拓扑电极位置图如上C图所示。电极位置的拓扑保持和非拓扑保持投影：**A） 使用非拓扑保持简单正交投影的电极位置的2-D投影。B） 电极在原始三维空间中的位置。C） 使用拓扑保持方位等距投影的电极位置的2-D投影**。可以这样简单的去理解：B中所展示的是脑电帽电极中的空间形状，而A表示的是简单的2-D投影落下的电极位置，C图表示使用AEP后的2-D投影电极位置。

#### 频域特征

:+1:采用“CloughTocher”方案(这是一种经典的脑电功率谱密度插值算法)对头皮上的散射功率测量值进行插值**(插值的原因是因为：64电极只能是64个孤立的头皮电极位置，而我们想要得到的是覆盖整个头皮层的功率值)**，并估算32×32网格上电极之间的值。对于每个感兴趣的频带重复该过程，从而产生对应于每个频带的三个地形活动图(α，β，θ)。然后将三个空间地图合并在一起，形成具有三个（颜色）通道的图像。此三通道图像作为深度卷积网络的输入，如以下部分所述。

:exclamation:这一部分电极投影、功率插值、活动图合并等操作过程在tf的代码中非常详细，pytorch代码中没有这个过程。

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding-Cognition/images/image-20230203153948980.png" alt="image-20230203153948980" style="zoom:67%;" />

:point_right:流程简述：**1.获取多个电极位置的时间序列信号 2.对每一个电极位置的信号在特定的频段内进行傅里叶变换提取功率谱，这里的特定频段分别为Theta，alpha，beta，得到的maps图中的每一个点都代表对应电极位置上的功率值 3.然后将这三个频段得到的maps拼在一起成为一张图，也就成为了一张三通道的图 EEG images**

:point_up:这里作者提出了一种思路： 1） 单帧方法：在整个试验期间(3.5s)，通过光谱测量构建单个图像，实际上这个单个图像是3.5s数据进行了除以7的平均。然后将构建的图像用作ConvNet的输入。2） 多帧方法：将每个试验分为0.5秒窗口，并在每个时间窗口上构建图像，每次试验提供7帧(总共3.5s)。然后将图像序列用作递归卷积网络的输入数据。作者在论文中强调了使用单帧方法的目的是为了寻找一种最高效的ConvNet结构，因此不需要时间信息(除以7做了平均就不存在时间信息了)，便将3.5s的数据构建单张图像。在多帧方法中使用单帧方法中寻找到的性能最好的ConvNet结构，并且在跨帧共享参数，这样操作的好处在于能够寻找到不包含时域信息时最佳的cnn架构，当在多帧模型中，我们将单帧的最佳cnn模型直接拿来用，需要时间信息时我们再利用lstm或者其他的结构即可。

#### 网络模型

##### 单帧模型

:pray:不考虑时间信息，直接将3.5s的数据做平均，模型的参数与结构如下：

![image-20230203155955220](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding-Cognition/images/image-20230203155955220.png)

:muscle:作者在论文中提到：经过他们的实验，D模型架构的是最好的，所以在多帧模型中跨参数使用D架构。这里需要说明的是：尽管这里的D架构也并不是非常复杂，与A、B、C架构比起来也仅仅是模块堆叠，但是还是那句话这是2016年所进行的工作，能有这样的消融比较已经非常不错了

##### 多帧模式

:v:由于多帧模型需要包含时域信息，因此作者提供了4种模型结构：



1）ConvNet+Maxpooling 

2）ConvNet+Temporal  convolution

 3）ConvNet+Lstm 

4）ConvNet+Mix(混合的)，



:cyclone:其中Mix包括Lstm和Temporal  convolution两种结构，我看代码的话py和tf的复现里面都是先让这两种结构单独执行，然后在最后的全连接层之前使用cat拼接来将两种结构的输出拼接起来，最后进行总的输出。

:cyclone:采用LSTM捕捉ConvNet激活序列的时间演化。由于大脑活动是一个时间动态过程，帧之间的变化可能包含有关潜在心理状态的附加信息。

:cyclone:模型的最终结果大致如下，列出了Mix，1D-Cov和Lstm的结果如下，其实并不能通过这个结果来判断出哪一种模型更适合这个认知实验的过程，毕竟要考虑到的是这是一篇16年的文章，那时候的深度学习模型没那么好。

![image-20230203161144754](https://cdn.jsdelivr.net/gh/lwlBCI/EEG-Decoding-Cognition/images/image-20230203161144754.png)

### 关于代码

:speech_balloon:关于文章的复现整个github中存在着诸多版本，除原版本外，**就我而言pytorch和tf的这两个版本是最贴合论文的处理过程的**

[![](https://img.shields.io/badge/pytorch-%40VDelv-brightgreen)](https://github.com/numediart/EEGLearn-Pytorch)

:skull:这个版本中存在的问题是：**数据预处理代码存在问题+公式顺序不对**，但在我个人上传的代码文件中这些问题均已修改，有兴趣的小伙伴可以到作者[![BXL](https://img.shields.io/github/followers/vdelv?label=VDelv&style=social)](https://github.com/VDelv/vdelv.github.io)中进行查看

[![](https://img.shields.io/badge/tensorflow-%40YangWangsky-brightgreen)](https://github.com/YangWangsky)

:thought_balloon:这个版本存在的问题是：**tensorflow的版本过低，1.x的版本，代码写的过于冗杂，如果想运行的话需要修改很多1.x语法变为2.x语法**，同样的在我个人上传的代码文件中这些问题均已修改，有兴趣的小伙伴可以到作者[![BXL](https://img.shields.io/github/followers/YangWangsky?label=YangWangsky&style=social)](https://github.com/VDelv/tf_EEGLearn)中进行查看

:love_letter:当然，文章的原作者在此[![BXL](https://img.shields.io/github/followers/pbashivan?label=pbashivan&style=social)](https://github.com/pbashivan)

😂:另外，关于所需要用到的数据，上面两位作者的仓库中都有，由于上传的文件大小和时间问题我就不上传了，大家可以去自行下载，如果实在实在不想去可以联系我获取！

:heart_eyes:感谢这三位作者所提供的代码及数据！

## 总结

:yum:总的而言，这篇论文提供了一种研究脑电信号的新方式，就是总结时域、频域和空间三个方面的信息输入到网络中，先寻找单帧结构模型作为基础模型部分，再根据时域信息寻找适合的lstm或者是一维卷积等结构，提出了多种模型结构供我们参考。但是话又说回来，我是真没看懂数据是怎么预处理的(包括插值和投影的那部分)，实在太复杂了，如果有兴趣的小伙伴深入研究后，可以告诉我怎么做的，我将不胜感激！！！

## Cite

Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

