# 关于**CCEP ECoG dataset across age 4-51**数据集中电极位置的一些说明

数据来源：[CCEP ECoG dataset across age 4-51 - OpenNeuro](https://openneuro.org/datasets/ds004080/versions/1.2.4)

文章来源：[Developmental trajectory of transmission speed in the human brain | Nature Neuroscience](https://www.nature.com/articles/s41593-023-01272-0)

相关代码：[MultimodalNeuroimagingLab/mnl_ccepAge: Repository to publicly share code to analyze/visualize CCEP data in BIDS (github.com)](https://github.com/MultimodalNeuroimagingLab/mnl_ccepAge)


> [!IMPORTANT]
> **Conclusions:**
> - **该数据集的电极位置配准到了fsaverage的空间（虽然文章中说被配准到了MNI152空间）**
> - **fsaverage空间实际上和MNI空间相同（近似相同）**
> 
> **因此，在使用中我们不需要对数据做额外的空间变换。**



我们在使用该数据集进行流形重建时，发现电极的位置坐标不规则，如下图所示

<img src="pic\微信图片_20240418164412.png" width="500px" style="zoom:40%;" /><img src="pic\微信图片_20240418164345.png" width="500px" style="zoom:40%;" />

因此，我们联系了通讯作者[Dora Hermes](http://orcid.org/0000-0002-8683-8909)，得到了如下回应：*“电极位置是准确的，而这些电极位置“不规则”的原因是因为MNI位置的提取方式。基于CT和MRI的配准，我们在每个患者中定位电极，然后使用Freesurfer的基于表面的归一化来获得MNI空间中的电极。因此，电极可以与Freesurfer的fsaverage表面结合使用。”*

而文章中的说法是：*“个体受试者的电极位置被转换为蒙特利尔神经学研究所(MNI)152空间”*。

<img src="pic\微信图片_20240417083228.jpg" alt="微信图片_20240417083228" width="400px" style="zoom: 35%;" /><img src="pic\image-20240418183237509.png" alt="image-20240418183237509" 
width="500px" style="zoom:80%;" />

因此，结合邮件、论文中的描述、以及一些其他公开的信息，我们得到了如下结论：

## 1. 该数据集的电极位置配准到了fsaverage的空间

我们从[公开的代码](https://github.com/MultimodalNeuroimagingLab/mnl_ccepAge/blob/master/scripts/makeFig1A_and_SupFig9_plotMNI.m)中发现，他们可视化时使用的是fsaverage的大脑皮层信息，如下图：

<img src="pic\image-20240418183542226.png" alt="image-20240418183542226" width="1000px" style="zoom:70%;" />

<img src="pic\image-20240418150823680.png" alt="image-20240418150823680" width="800px" style="zoom:70%;" />

并且[Dora Hermes](http://orcid.org/0000-0002-8683-8909)的回复邮件中也提到了：*“基于CT和MRI配准，在每个患者中定位电极后，使用了Freesurfer的基于表面的归一化来获得MNI空间中的电极位置。因此电极可以与Freesurfer的fsaverage表面结合使用”*。

因此，我们认为该数据集的电极位置配准到了fsaverage的空间中。后续我们会通过画图来验证这个说法。

## 2.fsaverage空间实际上和MNI空间相同（近似相同）

文章中提到了“为了可视化，个体受试者的电极位置被转换为蒙特利尔神经学研究所(MNI)152空间”，因此我们猜测：fsaverage空间就是MNI152空间（因为在代码中实际使用的是fsaverage）。

在[About the MNI space(s) – Lead-DBS](https://www.lead-dbs.org/about-the-mni-spaces/)这篇文章里，有关于不同模板的一些相关介绍：

<img src="pic\image-20240418184407878.png" alt="image-20240418184407878" width="800px" style="zoom:80%;" />

从中我们可以得知：MNI305是第一个MNI空间中的模板，MNI152模板是配准到了MNI305模板的空间中，因此MNI305模板和MNI152模板的空间是相同的MNI空间。

并且文章中明确指出了：*“在Freesurfer中，fsaverage空间已大致/近似地与MNI空间共同配准（更多详细信息请参阅[此处](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems)，感谢Denise Ruprai），但模板是基于不同的扫描者的数据（据我所知，基于 Buckner40数据集，该数据集已成为[genomics superstruct repository](https://www.neuroinfo.org/gsp/)的一部分）。并且，Thomas Yeo的实验室发表了一篇关于MNI和freesurfer空间之间准确转换的[论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239990/)。”* 如下图所示：

<img src="pic\image-20240418153236708.png" alt="image-20240418153236708" width="800px" style="zoom:75%;" />

<mark> **因此我们可以得到如下结论: MNI305和MNI152是MNI空间中的不同模板（不同扫描者平均的结果），fsaverage所在空间和MNI空间近似相同，因此，我们可以大致认为fsaverage的空间就是MNI305、MNI152的空间。但因为fsaverage并不是一个公认的空间，所以在需要专业描述的时候，文章中选择使用MNI152空间来描述fsaverage空间。** </mark>


> 下面给出了fsaverage空间和MNI152空间的准确转换：
> 
> <img src="pic\微信图片_20240418164430.jpg" alt="微信图片_20240418164430.jpg" width="800px" style="zoom:50%;" />
> 
> <img src="pic\微信图片_20240418164435.jpg" alt="微信图片_20240418164435" width="800px" style="zoom:50%;" />







一些相关链接：

[大话脑成像之十三：浅谈标准空间模板和空间变换 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/35662026)

[FsAverage - Free Surfer Wiki (harvard.edu)](https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage)

[FreeSurfer MRI reconstruction — MNE 1.7.0.dev187+g637b4343f documentation](https://mne.tools/dev/auto_tutorials/forward/10_background_freesurfer.html)

[Datasets Overview — MNE 1.7.0.dev187+g637b4343f documentation](https://mne.tools/dev/documentation/datasets.html#fsaverage)

[mne.datasets.fetch_fsaverage — MNE 1.7.0.dev187+g637b4343f documentation](https://mne.tools/dev/generated/mne.datasets.fetch_fsaverage.html#mne.datasets.fetch_fsaverage)

[About the MNI space(s) – Lead-DBS](https://www.lead-dbs.org/about-the-mni-spaces/)

[Accurate nonlinear mapping between MNI volumetric and FreeSurfer surface coordinate systems - PMC (nih.gov)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6239990/)



[CCEP ECoG dataset across age 4-51 - OpenNeuro](https://openneuro.org/datasets/ds004080/versions/1.2.4/file-display/CHANGES)

[MultimodalNeuroimagingLab/mnl_ccepAge: Repository to publicly share code to analyze/visualize CCEP data in BIDS (github.com)](https://github.com/MultimodalNeuroimagingLab/mnl_ccepAge)

