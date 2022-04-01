# my-weekly-paper-reading

<b>2022-04-27</b><br>

<i>Title</i>: <a href="https://arxiv.org/pdf/2203.15319.pdf">Can NMT Understand Me? Towards Perturbation-based Evaluation of NMT Models for Code Generation</a> (ICSE 2022 Workshop paper)<br>
<i>Author</i>: Pietro Liguori, Cristina Improta, Simona De Vivo, Roberto Natella, Bojan Cukic, Domenico Cotroneo<br>
<i>Comments</i>:<br>
Survey了一下在翻译任务（或者更广泛地说，Seq2seq任务）上是怎么定义Adversarial Attack的。Attack在NLU任务上可以定义得比较清楚，因为模型的输出是明确的label，如果我们的perturbation让模型输出的label改变了就可以被认为是攻击成功了。但是Seq2seq任务上模型的输出也是一个sequence，这个时候就比较难定义什么时候“模型的输出改变了”。这篇文章采用了多种semantics similarity评价指标，可以被简单分为两类：Automatic metrics和Manual metrics。Automatic包含了BLEU和Exact Macth（EM）以及一些基于子串分析的方法，如最长公共子串、编辑距离。这些都是直接和case的reference计算，如果经过perturb后模型的输出和reference计算的这几个分数有所下降就表明我们的attack有所影响。

本来我猜想他会自己定义一个threshold————这些metrics下降超过百分之多少就认为是攻击成功，但是他并没有。考虑到我们最后还是要在所有seed data上给出一个attack success rate来衡量各种不同攻击方法的performance，对每个case都设置threshold判断这个case单独是否攻击成功就显得有些没有必要。因此，本文直接考虑这几个metrics在原seed data上的score和在攻击后data上的score的差值来衡量不同攻击方法的performance。

除了这些Automatic metrics以外本文还包括了Manual的Semantic和Syntactic评价。由于本文的Seq2seq任务是natural language to code，syntactic评价的是生成的是不是可执行不报错的code（这步为什么不直接用一些编译器做自动化？或者一些静态分析方法也可以实现自动化，感觉用Manual有点不必要）；semantic评价的是natural language描述的是不是这段code做的事情。本文提出1. Unseen synonyms替换；2. 删去和target language相关的words；3. 删去variable或function的name。实验结果表明NL2Code模型基本在面对这三种情况都会fail。

<b>2022-03-27</b><br>

<i>Title</i>: <a href="https://arxiv.org/pdf/2203.09435.pdf">Expanding Pretrained Models to Thousands More Languages via Lexicon-based Adaptation</a> (ACL 2022)<br>
<i>Author</i>: Xinyi Wang, Sebastian Ruder, Graham Neubig<br>
<i>Comments</i>:<br>
Junjie Hu的那篇<a href="https://arxiv.org/abs/2003.11080">XTREME</a> (ICML 2020)表示，多语言pretrained models（MPLM）的性能极度依赖某种language的单语/parallel text data，就是说在训练的时候出现过的语言在下游任务性能会比没出现过的高很多（注意这边区别于上周present的<a href="https://arxiv.org/pdf/2203.08430.pdf">Cross-Lingual Ability of Multilingual Masked Language Models: A Study of Language Structure</a>，虽然这篇说MPLM性能不依赖parallel data，但是没有否认MPLM还是需要至少单语的数据训练），但是世界上还有大约7000种languages中还有大部分没有被覆盖到（mBERT只cover了1%）。怎么降低MPLM对这种形式的text data的依赖呢？这篇文章系统性地研究了双语词库（bilingual lexicons，就是双语对照辞典，并不是成型的textual data）的作用。本文回答了如何用双语词库生成textual/labeled data，这些data怎么与已有的data相结合？

本文用了一个看起来有点笨的办法：在有text data的语言上取一个sentence，然后根据双语词库做word-to-word translation来生成target的pseudo text data。然后用MLM objective + translated text在原text上的label的cross entropy两个loss做训练。本文还用了distillation来优化generated data。这个笨办法有几大问题：1. 词库有可能很多词没有收录，导致翻译后还是有很多原语言的words；2. 没有考虑一词多义（morphological）；3. 没有考虑语法顺序（主谓宾或主宾谓这种）。但是实验结果还是表明他有不错的性能，我觉得可以用上周present的<a href="https://arxiv.org/pdf/2203.08430.pdf">Cross-Lingual Ability of Multilingual Masked Language Models: A Study of Language Structure</a>的一个发现解释：MPLM的cross-lingual性能不太依赖语言的constituent order和word co-occurrence。

<i>Title</i>: <a href="https://arxiv.org/pdf/2203.10753.pdf">Match the Script, Adapt if Multilingual: Analyzing the Effect of Multilingual Pretraining on Cross-lingual Transferability</a> (ACL 2022)<br>
<i>Author</i>: Yoshinari Fujinuma, Jordan Boyd-Graber, Katharina Kann<br>
<i>Comments</i>:<br>
多语言pretrained models（MPLM）在zero-shot场景上可以表现的很好，但是其中的很多机理都还没被搞清楚。我们知道Conneau在他的<a href="https://aclanthology.org/2020.acl-main.747.pdf">XLM-R</a>中提出过“curse of multilinguality”，当训练时用到的languages数量上升到一定阶段后，下游任务上（非zero-shot transfers）language的performance会开始下降。但是这个发现并没有探索下游任务上unseen的languages（transfer能力）会有什么影响，本文就旨在填补这个gap，具体来说提了三个research questions：
1. 预训练时用的languages数量有什么影响？<br>
2. 在使用了model adaptation下问题1的情况如何？<br>
3. 如果用的languages是related的（我的理解是同family），问题1的情况又如何？

本文实验设置在XTREME的POS tagging和XNLI上。结论：
1. 和“curse of multilinguality”的结论不同，在unseen language上的性能并没有观察到一个明显的drop趋势，基本上维持上升（虽然上升幅度很小）；
2. 在continued pretraining后，总体上accuracy比不用的情况更高，并且增长也更明显；
3. 使用的language是不是related的基本没有影响。

<b>2022-03-20</b><br>

<i>Title</i>: <a href="https://arxiv.org/pdf/2203.08430.pdf">Cross-Lingual Ability of Multilingual Masked Language Models: A Study of Language Structure</a> (ACL 2022)<br>
<i>Author</i>: Yuan Chai, Yaobo Liang, Nan Duan (MSRA)<br>
<i>Comments</i>:<br>
来自MSRA的paper。讨论的核心问题是：mBERT和XLM-R在zero-shot cross-lingual任务上表现出了极高的性能，但是他们的训练都没有使用cross-lingual的supervision或者aligned（parallel） data，那么他们cross-lingual的能力到底是怎么来的呢？本文给出的回答是：语言之间存在共性。具体来说，本文研究了三种属性：1. constituent order（即谓宾顺序、冠词名词顺序和介词名词顺序）；2. composition（语法树）；3. word co-occurrence。具体研究方法为通过构造一个语言来test这三种property，每次remove一个property然后测XNLU性能。本文通过交换三种constituent顺序来去除property 1；通过shuffle语法树中每层siblings的顺序来去除property 2；shuffle所有word得到的是仅保留property 3的bag of words model。结论：1的影响小（在XNLI上性能差距仅1%），2的影响在本文study的两个任务（entailment和sentence retrieval）上较大。

这个问题的讨论也有一段时间了，<a href="https://aclanthology.org/P19-1493/">How Multilingual is Multilingual BERT?</a>和<a href="https://aclanthology.org/D19-1077/">Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT</a>说是因为source和target存在相同的words，然后从中学到transfer的信息；但是<a href="https://aclanthology.org/2020.acl-main.536/">Emerging Cross-lingual Structure in Pretrained Language Models</a>和<a href="https://openreview.net/forum?id=HJeT3yrtDr">Cross-Lingual Ability of Multilingual BERT: An Empirical Study</a>又发现两个来自不同domain（family）的语言也能zero-shot transfer的很好，所以不是这个原因。

本文的缺点可能是没有落脚到improvement上（他们说把findings应用到improving performance是他们的future work）；他们人工构造的语言拿来做实验感觉也不是十分convincing（只拿了English来魔改）。

<i>Title</i>: <a href="https://arxiv.org/pdf/2203.09326.pdf"></a>Combining Static and Contextualised Multilingual Embeddings (ACL 2022)<br>
<i>Author</i>: Katharina Hämmerl, Jindrich Libovický, Alexander Fraser<br>
<i>Comments</i>:<br>
这是2022年ACL接受的正会短文，一个简单有效的方法。mBERT和XLM-R在zero-shot cross-lingual任务上表现出了极高的性能，但是他们的language-neutrality（即语言之间align得如何）不太行，topologically distent的languages之间就没有很aligned的representations，这样transfer效果会不好。本文注意到，在contextual representation出来之前的static representation可以被有效地align在一起，但是相比contextual representation他们又不那么expressive。本文希望结合static和contextual representation以获得更好的representation去做下游的需要transfer的tasks。

本文的训练方法不需要parallel data。具体方法很简单：第一步，先拿很多个monolingual data训很多个XLM，得到对应的representation（denoted by X2S-M）；第二步，根据很多previous work，这些static representation能简单的通过线性变换align起来，于是就过一个Mapping模块（VecMap）得到aligned static representation（denoted by X2S-MA）；第三步，在做正常的multilingual contextual representation训练时，多加一个和X2S-MA靠近的loss（用MSE或者Deep CCA）。本文在QA（XQuAD、TyDuQA-GoldP）、Sequence Labelling（PAN-X、UD-POS）和Sentence Retrieval（Tatoeba）上做了实验，baseline选择了直接拿fasttext representation来做DCCA的训练。实验结果发现X2S-MA+DCCA效果最好，但是如果用MSE在Tatoeba数据集上performance只有惊人的10.05（别人都是50-60），作者也没解释为什么。

<b>2022-03-13</b><br>

<i>Title</i>: <a href="https://proceedings.neurips.cc/paper/2019/file/c04c19c2c2474dbf5f7ac4372c5b9af1-Paper.pdf">Cross-lingual Language Model Pretraining</a> (NeurIPS 2019)<br>
<i>Author</i>: Alexis Conneau, Guillaume Lample (FAIR)<br>
<i>Comments</i>:<br>
最近开始着手survey cross-lingual natural language understanding (XNLU) 的工作，先从领域内经典的文章开始。从XNLI这个dataset开始，发现一系列XNLU的工作都是出自Conneau之手，这周的paper reading就带来两篇他的工作，分别是这一篇NeurIPS 2019的和下一篇ACL 2020的。

这篇工作属于XNLU领域citation第一梯队的paper之一（就是有名的XLM）。简单讲，这篇paper主要针对只有单语的情况和有parallel data的情况分别提出两种unsupervised和一种supervised的pre-training方式。第一种unsupervised：CLM，是最常见的given previous t-1 words去model the t-th word的LM建模方式；第二种unsupervised：MLM，是用和BERT一样的masking策略来建模（randomly mask 15%）；第三种supervised的方式TLM是把两个parallel data直接concat在一起然后randomly mask tokens。作者提到这个方式可以enable模型在一个语言被mask缺少information的时候转而去看另一个语言，以此增强alignment。

实验部分作者有几个发现：1. XLM可以为zero-shot XNLU任务提供更好的encoder（实验做在XNLI dataset）；2. XLM可以作为更好的unsupervised和supervised NMT method（实验做在WMT‘16 dataset）；3. XLM可以增强低资源语言的学习；4. XLM可以unsupervised地学到cross-lingual的word emb。值得注意的是在单语实验的baseline里面，作者使用了额外的machine translation模型来测试cross-lingual性能。

<i>Title</i>: <a href="https://aclanthology.org/2020.acl-main.747.pdf">Unsupervised Cross-lingual Representation Learning at Scale</a> (ACL 2020)<br>
<i>Author</i>: Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzman, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov (FAIR)<br>
<i>Comments</i>:<br>
这一篇同样来自Conneau的工作XLM-RoBERTa (XLM-R)，收录在ACL 2020。本文提出的模型在classification，sequence labeling和QA任务上都达到SOTA。本文通过empirical study发现了总languages数量和每个language的capacity之间存在tradeoff（curse of multilinguality），high和low resource languages的performance之间存在tradeoff。最后，本文发现多语言模型并不一定要牺牲某些语言的performance来提升整体性能。

相比前作XLM，本文没有使用language embeddings，使得模型可以处理code switching的情况；本文使用CommonCrawl进行训练而非Wikipedia，语言数量和规模得到大幅提升。实验部分，作者在多语言数据集上同时测试了多语言模型和单语模型（例如GLUE上用RoBERTa对比XLM-R）；此外，作者还发现mBERT和XLM-100在建模低资源language上不如XLM-R的原因是mBERT和XLM-100太过于依赖cross-lingual transfer。感觉在这个问题上可以引入上周paper reading我记录的ICLR关于fine-tuning会导致feature distortion的paper。有可能在corss-lingual的transfer上也发生了类似的feature distortion。

<b>2022-03-06</b><br>

<i>Title</i>: <a href="https://arxiv.org/pdf/2202.10054.pdf">Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution</a> (ICLR 2022 Oral)<br>
<i>Author</i>: Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, Percy Liang (Stanford)<br>
<i>Comments</i>:<br>
这篇工作是Stanford的Percy Liang组的工作，他们组早年focus在adversarial attack and defense比较多，因此我有一直在follow。最近一两年他们关注adversarial比较少了，倒是转向关注interpretability，我觉得这篇paper他们给一个可解释性的任务找到了不错的实际落脚点，就是现在越来越多的real-world scenario需求在OOD上也要取得不错的性能，因此需要模型不仅能在ID data上能泛化并且还要在OOD data上也能泛化。有了这个落脚点，他们就可以切入到为什么finetune会distort pretrained feature这样的观察与分析上来。

有两种方法把预训练模型应用到下游任务：全参数一起finetune；只finetune head。这篇文章发现只finetune head会在OOD data上取得更高accuracy。这篇文章从理论上证明了一个两层的linear feature extractor + linear head会存在ID和OOD accuracy上的tradeoff，并且本文发现这种tradeoff并不能被early stopping所取消。最后，本文建议采用二阶段（先只finetune head再finetune全参数）以在ID和OOD data上都获得提升。

基本的想法是：首先，预训练学习到了general的feature，包含在模型参数中。然后当我们finetune到某个下游任务时，如果一开始就所有参数一起训会导致只有下游任务in domain的feature进行了更新，which causes OOD feature被遗忘。

有用的数学技巧：1. 用旋转矩阵定义feature extractors之间的distance；2. 与训练数据空间正交的方向会在和梯度做内积的时候得0；3. 把ID space和OOD space看作两条坐标轴，只finetune head不会改变feature extractor在ID坐标轴的位置，但是全参数finetune会改变。实验部分选取了10个包含ID和OOD的datasets，有些domain shift大有些小，在实验中也观察到ID和OOD的accuracy确实存在tradeoff，但是使用本文推荐的two-stage finetune方法就可以减轻这个问题。

本文的实验选择的都是CV上的数据集，我觉得有迁移到NLP的潜力。

<i>Title</i>:<a href="https://aclanthology.org/2021.emnlp-main.126.pdf">Improving Zero-Shot Cross-Lingual Transfer Learning via Robust Training</a> (EMNLP 2021)<br>
<i>Author</i>: Kuan-Hao Huang, Wasi Uddin Ahmad, Nanyun Peng, Kai-Wei Chang<br>
<i>Comments</i>:<br>
本文是来自UCLA Kai-Wai Chang组的工作。Kai-Wai Chang与UCLA组的谢卓叡老师有密切合作，他们一直关注adversarial attack以及defense，因此他们的lab也是我一直follow的。他们的研究领域其实很广泛，从certified robustness到representation learning中的可解释性工作都有所涉及。最近他们也在扩展adversarial attack和defense到更广泛的任务上，探究在不同任务下的不同可能性。因为我之前一直在做传统分类任务上的adversarial attack，因此之后在这个组的方向也可能会是翻译里的问题和testing（adversarial attack与defense）的结合，这篇paper是一个不错的视角，也是一个不错的adversarial在实际应用中的落地方案。

要求在多语言下达到word和phrase级别的alignment需要有大量word-level parallel的训练data，这在low-resource语言中不可行。另一种方法就是训练encoder使之有能力抵御在low-resource下游任务训练时可能会遇到的noice（没有well-align），换言之就是使之具有高zero-shot performance。具体的方法在adversarial领域已经十分成熟：adversarial training以及random smoothing。

文祥学长在他的review中提到，前人的工作都是在拉近representation距离，而这篇工作是在把某个语言的容错空间（robust region，或者理解为decision boundary）扩大。我在这里再补充一个和前人工作的不同之处：前人很多工作在处理不同语言的representation时，会把每个语言看作独立的subspace，然后假设每个subspace包含相似的结构，然后不同语言之间就可以通过线性变换映射。而这篇工作是把每个语义看作独立的subspace，这个空间里包含不同语言的对这个语义的表达。我觉得一个有意思的future work是探究这每一个语义空间中不同语言之间的结构是不是存在共性。更进阶一点的猜想是一般矩阵可以分解为行空间和列空间，multi lingual representation space是否也可以分解为语义空间和语言空间，从而探索更多representation上的性质？

By <i>Jen-tse Huang</i>
