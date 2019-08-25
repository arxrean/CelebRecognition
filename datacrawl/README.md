There have been several datasets for star face recognition such as IMDB and [MS Celeb](https://www.msceleb.org/).  However,  the stars in it can be a little out of date. Besides, each star has only few samples which may not cover all living and movie styles. Therefore, I create a star dataset in which part of stars have dozens of images while the others have just one image for each.  Most stars are well known as least in 2019 year..I will record the process of creating it in the following. All codes are available in my Github.

# Data Crawl

I firstly crawled star information and their images from [happyjuzi.com](http://www.happyjuzi.com/) website and many other small websites.

## Star Info

I crawled each star based on their unique url id and obtained several properties that may help in the future model training, as denoted in the following table.

| Property   | Description                           |
| ---------- | ------------------------------------- |
| id         | the unique id for each star           |
| name       | the original name that may be Chinese |
| e_name     | English name if they have             |
| star_type  | actor/(Internet) celebrity/others     |
| star_other | other info like their sign            |
| area       | country they are from                 |
| desc       | a few sentences to describe them      |
| reels      | their works                           |
| gender     | 1 for male while 0 for female         |

## Star Images

The images consists of two parts. The first part is obtained from image repository of each star while the second part is from their profile pictures, which is denoted in the following image.

![alt text](https://res.cloudinary.com/dhortnuos/image/upload/v1566355729/blog/star-dataset/1_ggikv6.png "Figure_1")

The key images are much more likely to be true images for corresponding stars compared with other images, since many noisy or even unrelated images will be uploaded by users.  The following table illustrates basic info for my original dataset.

| Star  | Images | Key   |
| ----- | ------ | ----- |
| 13715 | 224438 | 13322 |

We can see most stars have their key images. The distribution of the number of images per each star is shown in the following figure.

![alt text](https://res.cloudinary.com/dhortnuos/image/upload/v1566440487/blog/star-dataset/original_dataset_hist_txhfim.png "Figure_1")

# Data Preprocess

Since a large part of scrawled data are just unrelated images with targeted stars, we need to preprocess data in several steps.

## Remove similar Images

Different fans for one star may upload a same image. I use pHash to hash each image per star and compare each two image hashes, calculating their euclidean distance. If their similarity is higher than the predefined threshold, I will mark one of them as the duplicate and remove it.

## Remove group stars

There are some groups which consists of several or many stars. But since each single star will be also recorded in many website, these group labels will influence the training. So I just remove these group stars. 

I use pretrained MTCNN model to extract faces in Key Images. Since these images always only contain the single star face if the star is not a group. I set the threshold for MTCNN very high and remove the star if the model detect more than one face.

## Extract Faces

I extract faces in each image, if there are more than two faces in one image, I just remove it. If there are two faces, I check it and choose the right one. The face is $112\times 112$  gray and aligned by landmarks.

## Remove Similar or Unrelated  Faces

For each star, there may exist same face image or other faces which are not the targeted star.

I first use pretrained lightCNN to extract 256-d features for each face. Then I use both euclidean distance and cosine similarity to calculate the distance between two faces. If the value is higher than $thd1$ or lower than $thd2$, I remove one of the raw face image. Please note the dataset is very sensitive to $thd1$ and $thd2$. Therefore I need to make action only if there is a very high confidence.

Then I finetuned a gender classification model based on crawled data. The model check each raw face. If a face image is recognized as wrong gender with high score, I just remove it. 

## Result

The following table is the distribution of all stars.

| Total Images | Total stars |
| ------------ | ----------- |
| 80062        | 9233        |

And the number of images per star is in the following figure:

![alt text](https://res.cloudinary.com/dhortnuos/image/upload/v1566457293/blog/star-dataset/face_dataset_hist_mollje.png "Figure_1")

More infomation for model training and inference can be seen in my future article. The whole data can be seen in this [Baidu drive link](https://pan.baidu.com/s/19RNnOFrCiB7Bw5fMxb3MyA).