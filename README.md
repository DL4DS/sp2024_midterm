# DS598 DL4DS Midterm Project

For my project I used the Microsoft Git Large model trained on the coco image dataset [4][5]. I found that this one was relatively simple to implement and work with. 
Fine tuning the model took the most time, I had to experiment with the attention mask, learning rate, and batch sizes to finally get a model that performs well. I
ended up finding a nice parameter set that got me a CIDEr score of ~75 after only 1 epoch. I had fun learning about hugging face and implementation of deep learning models!


## References

1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space
4. [Microsoft Git Large](https://huggingface.co/microsoft/git-large-coco)
5. GIT: A Generative Image-to-text Transformer for Vision and Language, Jianfeng Wang and Zhengyuan Yang and Xiaowei Hu and Linjie Li and Kevin Lin and Zhe Gan and Zicheng Liu and Ce Liu and Lijuan Wang (2022)
   
