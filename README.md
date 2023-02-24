# Lapses
Lapses is a repository that aims to depict the progression of dementia using mathematics and machine learning. The repository contains an implementation of a deep convolutional generative adversarial network (DCGAN) that generates images depicting the progression of dementia, with the intention of destroying them during the generation process to represent the loss of cognitive function, i.e. with the end-goal of dementia.

## How it works
The deep convolutional generative adversarial network (DCGAN) uses convolutional neural networks to generate images. In the context of Lapses, the DCGAN is used to generate a series of images that depict the progression of dementia, with each image representing a stage in the disease's progression.

The real images used in the training set were obtained by scraping 10,000 images from [thispersondoesnotexist.com](thispersondoesnotexist.com). These images are computer-generated faces that do not represent any real individuals.

## Using it
The implementation of the DCGAN is currently a Jupyter notebook in Python and is still a work in progress. Once completed, users will be able to use the notebook to train the DCGAN on the provided dataset and generate images that depict the progression of dementia.
