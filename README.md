
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
  📄 <a href="https://arxiv.org/abs/2412.17799">[PDF]</a> |
  <!-- 💻 <a href="Google Colab">[Google Colab (coming soon)]</a> | -->
  💻 Google Colab (coming soon)
</p>

[Akarsh Kumar](https://x.com/akarshkumar0101) $^{1}$ $^2$, [Chris Lu](https://x.com/_chris_lu_) $^{3}$, [Louis Kirsch](https://x.com/LouisKirschAI) $^{4}$, [Yujin Tang](https://x.com/yujin_tang) $^2$, [Kenneth O. Stanley](https://x.com/kenneth0stanley) $^5$, [Phillip Isola](https://x.com/phillip_isola) $^1$, [David Ha](https://x.com/hardmaru) $^2$
<br>
$^1$ MIT, $^2$ Sakana AI, $^3$ OpenAI, $^4$ The Swiss AI Lab IDSIA, $^5$ Independent

## Abstract
With the recent Nobel Prize awarded for radical advances in protein discovery, foundation models (FMs) for exploring large combinatorial spaces promise to revolutionize many scientific fields. Artificial Life (ALife) has not yet integrated FMs, thus presenting a major opportunity for the field to alleviate the historical burden of relying chiefly on manual design and trial-and-error to discover the configurations of lifelike simulations. This paper presents, for the first time, a successful realization of this opportunity using vision-language FMs. The proposed approach, called *Automated Search for Artificial Life* (ASAL), (1) finds simulations that produce target phenomena, (2) discovers simulations that generate temporally open-ended novelty, and (3) illuminates an entire space of interestingly diverse simulations. Because of the generality of FMs, ASAL works effectively across a diverse range of ALife substrates including Boids, Particle Life, Game of Life, Lenia, and Neural Cellular Automata. A major result highlighting the potential of this technique is the discovery of previously unseen Lenia and Boids lifeforms, as well as cellular automata that are open-ended like Conway’s Game of Life. Additionally, the use of FMs allows for the quantification of previously qualitative phenomena in a human-aligned way. This new paradigm promises to accelerate ALife research beyond what is possible through human ingenuity alone.

## Repo Description
This repo contains a minimalistic implementation of ASAL to get you started ASAP.
Everything is implemented in the [Jax framework](https://github.com/jax-ml/jax), making everything end-to-end jittable and very fast.

We have already implemented the following ALife substrates:
- [Lenia](https://en.wikipedia.org/wiki/Lenia)
- [Boids](https://en.wikipedia.org/wiki/Boids)
- [Particle Life](https://www.youtube.com/watch?v=scvuli-zcRc)
- Particle Life++
  - (Particle Life with changing color dynamics)
- [Particle Lenia](https://google-research.github.io/self-organising-systems/particle-lenia/)
- Discrete Neural Cellular Automata
- [Continuous Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Game of Life/Life-Like Cellular Automata](https://en.wikipedia.org/wiki/Life-like_cellular_automaton)

You can find these substrates at [models/](models/)

The main files to run ASAL are the following:
- [main_opt.py](main_opt.py)
  - Run this for supervised target and open-endedness
  - Search algorithm: Sep-CMA-ES (from evosax)
- [main_illuminate.py](main_illuminate.py)
  - Run this for illumination
  - Search algorithm: custom genetic algorithm
- [main_sweep_gol.py](main_sweep_gol.py)
  - Run this for open-endedness in Game of Life substrate (b/c discrete search space)
  - Search algorithm: brute force search

## Running on Google Colab
<!-- Check out the [Google Colab](here). -->
Coming soon!

## Running Locally
### Installation 

To run this project locally, you can start by cloning this repo.
```sh
git clone git@github.com:SakanaAI/asal.git
```
Then, set up the python environment with conda:
```sh
conda create --name asal python=3.10.13
conda activate asal
```

Then, install the necessary python libraries:
```sh
python -m pip install -r requirements.txt
```
However, if you want GPU acceleration (trust me, you do), please [manually install jax](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) according to your system's CUDA version.

### Running ASAL
Check out [asal.ipynb](asal.ipynb) to learn how to run the files and visualize the results.

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

