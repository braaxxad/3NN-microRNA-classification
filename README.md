# 3NN-microRNA-classification

## Motivation
> *“Machine Learning: field of study that gives computers the ability to learn without being clearly programmed”* **– Arthur Samuel 1959**

I wanted to use this idea and apply it in biology. More specifically, field of deep learning offers endless possibilities, literally anywhere; why not biology? As long as the data is served, it is **all you can eat, free-for-all, a line of buffet.** 

Although this program is **3-layer neural network**, it actually does not practice deep learning. Nevertheless, this gave me a great start in developing programs towards deep learning, utilizing a machine learning tool such as **Pytorch.**

## Summary
Using **48 features** of microRNA, this 3-layer neural network (built with Pytorch) identifies microRNA. The features include calculations of **1). sequence conservations, 2). 2-D structures, and 3). thermodynamic properties.** Preparing data of such calculations would require more time and coding, so I have acquired data that was already prepared by a previous study on microRNA classification [[1]](https://doi.org/10.1093/bioinformatics/btp107). The data includes: 691 human premature microRNAs (Why premature? [Check Note1](#extra-notes)) and 8494 human pseudo-hairpin RNAs ([Check Note2](#extra-notes)).

In this classification problem microRNAs are positive, and pseudo-hairpin RNAs are negative. However there is a big imbalance between numbers of positive samples and negative samples (691 to 8494). To solve this imbalance, I used a method of **under-sampling** the negative samples using **k-means clustering**. I used **Scikit-learn** to cluster 8494 pseudo-hairpin RNAs into 12 clusters (because negative to positive ratio was roughly about 12:1). Each cluster was combined with the 691 positive data (microRNA) to make 12 different dataset. Then each dataset was trained and tested for classification using **Scikit-learn SVM**. Ultimately, I chose 3 clusters that yielded highest accuracy in this sample SVM test (Reasoning: [Check Note3](#extra-notes)). The number of sequences in negative set was trimmed down to 693 sequences to precisely match the number of positive set. The dataset was saved in CSV file.

In python, I wrote a 3-layer neural network using libraries: **torch, pandas, and matplotlib**. The dataset was input using **pandas dataframe** (I just like using pandas; always have used for working with data). After randomizing the orders, the dataset was split into train, validation, and test sets(respectively 70:15:15 ratio). I split the training dataset again into 24 mini batches for smoother training process (I have an old laptop). I used dictionary objects for this step. Training ran for **45 epochs** with learning rate of **3e-4**. The NN architecture is made up of 128, 64, 2 neurons; in first, second, and third layer respectively (2 hidden, 1 output). Activation function **ReLU** was used in hidden layers, and normalization function **Softmax** was used in output layer. Everything afterwards is straightforward business as usual, so I am skipping the further detail. Attached below are the sample outputs.

### Output at the end of epoch 45
|  Epoch |  Training Loss  |  Validation Loss  |  Test Prediction Accuracy |
|:------:|:------:|:-----:|:-----:|
| 45/45 | 0.012 | 0.012 | 0.984 |

### Training Performance Plot
![image](https://github.com/braaxxad/3NN-microRNA-classification/blob/master/3NN-microRNA-classifcation-sample-output.png)

## What's Next?
I have already finished a Recurrent Neural Network that is able to learn from the raw RNA seqeunces, rather than the handcrafted features used in `this.project`. Another upside to using raw sequences is that it is available in virtually infinite quantity. Also, I did not have to do extra work on the data side, I just had to collect them from a database called [miRBase](http://www.mirbase.org/). They provide identified microRNAs across many species (both eukaryote and prokaryote).

I am currently editting and organizing the python code (again built with Pytorch). Only problem I have with this model is that my laptop does not have a GPU that is compatible for usage. So I was not able to take advantage of Pytorch Cuda for faster performance.

## Extra Notes
1. It is difficult to identify miRNAs directly because they are relatively very short (only ~20 sequences long), which means features to extract for classification is very minimal, or even almost impossible. Thus, most studies today focus on identifying premature miRNA (pre-miRNAs) rather the mature form. In life span of micro RNA, there are three stages in its cyle: primary -> premature -> mature. Identifying pre-miRNAs is easier in comparison to mature miRNAs because pre-miRNAs are much longer (~80 sequences) and this allows room for the molecules to form hairpin-loop structure with more structural features. [Back to summary](#summary)

2. Pseudo-hairpin RNAs are essentially meaningless RNAs that share similar 2-D structure as the real microRNAs, thus an actual classification problem. [Back to summary](#summary)

3. I chose the clusters with highest accuracy because I believe they yielded highest based on reasoning that they are statistically more differentiable from the positive data (microRNA). Because I am working with relatively small dataset for this classification, it would be in my best interest to design my dataset that would train my actual model towards ability to differentiate positive from negative as much as possible. [Back to summary](#summary)

**About note number 3: This project was done awhile ago. As I track back now, I see that maybe picking out each equal amount of data from the 12 clusters would be a better idea. This would better represent the whole dataset of 8494 pseudo-hairpin RNAs.**


## Reference

> 1.  R. Batuwith, and V. Palade. microPred: effective classification of pre-miRNAs for human miRNA gene prediction. Bioinformatics Vol 25 Issue 8 pages 989-995. (2009) https://doi.org/10.1093/bioinformatics/btp107
