# Contrastive Learning

My goal was to investigate and implement self-supervised learning techniques in the pre-training phase of image processing to enhance feature extraction capabilities of models applied to medical imaging. I decided to focus on techniques in **contrastive learning**. 

The idea is to create paired views of the same image using a stochastic data augmentation pipeline and then ensure that their representations in the embedding space are closer to each other than to representations of other images/instances ([SimCLR](https://arxiv.org/abs/2002.05709)). This model is implemented using a ResNet50 backbone as the base encoder, followed by a projection head that maps the encodings to an embedding space where contrastive loss is applied. 

### Weakly-Supervised Contrastive Learning (WCL)
To enhance feature extraction further (SimCLR ignores the relationship between similar images, regarding them as separate instances) introduce a second projection head ([WCL](https://arxiv.org/abs/2110.04770)). This projection head identifies similar instances and generates weak labels using the embedding vectors generated by the first, and uses it as a supervisory signal to pull them closer in the embedding space using a supervised version of the contrastive loss. 

### Medical Imaging

The goal is to use this technique to pre-train a domain-specific CNN-based model (e.g. X-Rays) using a large, publicly available dataset and evaluate performance benefit over Image-Net trained CNN model on another dataset in the same domain. 

### Status
- Data Augmentation using Albumentations
- DataSet that returns augmented views of (currently, CIFAR) images.
- Defined ResNet backbone, Projection Head, SimCLR model
- Training loop with Adam optimizer.
- Linear Evaluation pipeline (currently with CIFAR test images). 
