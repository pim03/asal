import argparse
import os
from collections import defaultdict
from functools import partial

import evosax
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
from jax.random import split
from PIL import Image
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0)
group.add_argument("--save_dir", type=str, default=None) # path to save results to

group = parser.add_argument_group("model")
group.add_argument("--substrate", type=str, default='boids') # substrate name

group = parser.add_argument_group("data")
group.add_argument("--sampling", type=int, default=1) # number of images to render during one simulation rollout
group.add_argument("--prompts", type=str, default="an artificial cell,a bacterium") # prompts to optimize for
group.add_argument("--clip_model", type=str, default="clip-vit-base-patch32") # clip-vit-base-patch32 or clip-vit-large-patch14 (don't touch this)
group.add_argument("--coef_prompt", type=float, default=1.) # coefficient for prompt loss
group.add_argument("--coef_softmax", type=float, default=0.) # coefficient for softmax loss (only for multiple temporal prompts)
group.add_argument("--coef_novelty", type=float, default=0.) # coefficient for open-endedness loss (only for single prompt)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1) # number of init states to run simulation for
group.add_argument("--pop_size", type=int, default=16) # population size for Sep-CMA-ES
group.add_argument("--n_iters", type=int, default=10000) # number of iterations
group.add_argument("--sigma", type=float, default=0.1) # mutation rate

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)  # set all "none" to None
    return args

def main(args):
    prompts = args.prompts.split(",")
    if len(prompts) > 1: # doing multiple prompts
        args.sampling = len(prompts)
    print(args)
    
    fm = foundation_models.create_foundation_model('clip')
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=substrate.rollout_steps, sampling=(sampling, True), img_size=224, return_state=False)

    rng = jax.random.PRNGKey(args.seed)
    z_txt = fm.embed_txt(prompts) # P D

    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=sim.n_params, sigma_init=args.sigma)
    es_params = strategy.default_params
    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data['z']

        if len(prompts) > 1: # doing multiple prompts
            loss_prompt = asal_metrics.calc_supervised_score(z, z_txt)
            loss_softmax = asal_metrics.calc_supervised_score_softmax(z, z_txt)
            loss = loss_prompt * args.coef_prompt + loss_softmax * args.coef_softmax
            loss_dict = dict(loss=loss, loss_prompt=loss_prompt, loss_softmax=loss_softmax, scores=scores)
        else: # doing single prompt
            loss_prompt = asal_metrics.calc_supervised_score(z, z_txt)
            loss_oe = asal_metrics.calc_open_endedness_score(z)
            loss = loss_prompt * args.coef_prompt + loss_oe * args.coef_oe
            loss_dict = dict(loss=loss, loss_prompt=loss_prompt, loss_novelty=loss_novelty)
        return loss, loss_dict

    @jax.jit
    def do_iter(es_state, rng):
        rng, _rng = split(rng)
        params, next_es_state = strategy.ask(_rng, es_state, es_params)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))
        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))  # mean over bs
        next_es_state = strategy.tell(params, loss, next_es_state, es_params)
        data = dict(best_loss=next_es_state.best_fitness, generation_loss=loss.mean(), loss_dict=loss_dict)
        return next_es_state, data

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)

        data.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())
        if args.save_dir is not None and (i_iter % (args.n_iters//10)==0 or i_iter==args.n_iters-1):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best)

if __name__ == '__main__':
    main(parse_args())
