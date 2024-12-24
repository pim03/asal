
<h1 align="center">
  <a href="https://sakana.ai/asal">
    <img width="600" alt="Discovered ALife Simulations" src="https://pub.sakana.ai/asal/assets/png/cover_video_square-min.png"></a><br>
</h1>


<h1 align="center">
Automating the Search for Artificial Life with Foundation Models
</h1>
<p align="center">
  📝 <a href="https://sakana.ai/asal">[Blog]</a>
  🌐 <a href="https://asal.sakana.ai/">[Paper]</a> |
  📄 <a href="https://arxiv.org/">[PDF]</a> |
  💻 <a href="Google colab">[Google Colab]</a> |
</p>

[Akarsh Kumar](https://x.com/akarshkumar0101) $^{1}$ $^2$, [Chris Lu](https://x.com/_chris_lu_) $^{3}$, [Louis Kirsch](https://x.com/LouisKirschAI) $^{4}$, [Yujin Tang](https://x.com/yujin_tang) $^2$, [Kenneth O. Stanley](https://x.com/kenneth0stanley) $^5$, [Phillip Isola](https://x.com/phillip_isola) $^1$, [David Ha](https://x.com/hardmaru) $^2$
<br>
$^1$ MIT, $^2$ Sakana AI, $^3$ OpenAI, $^4$ The Swiss AI Lab IDSIA, $^5$ Independent


## Installation 

First, recreate the conda environment for the project
```shell
conda env create -f environment.yaml
pip install -r requirements.txt
```

If you encounter any issues installing this environment, please [manually install jax](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) according to your system's CUDA version.
Then manually install the following libraries:
- flax==0.9.0
- transformers==4.45.2
- tqdm==4.66.5
- einops==0.8.0
- evosax==0.1.6
- imageio==2.35.1
- imageio-ffmpeg==0.5.1
- matplotlib==3.9.2
- pillow==10.4.0

## Running ASAL
Check out [asal.ipynb](asal.ipynb)

## Reproducing Results from the Paper
Coming soon!
  
## Bibtex Citation
To cite our work, you can use the following:
```
@article{kumar2024asal,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Akarsh Kumar and Chris Lu and Louis Kirsch and Yujin Tang and Kenneth O. Stanley and Phillip Isola and David Ha},
  year = {2024},
  url = {https://asal.sakana.ai/}
}
```

