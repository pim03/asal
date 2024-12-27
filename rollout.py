import jax
from jax.random import split

def rollout_simulation(rng, params, s0=None,
                       substrate=None, fm=None, rollout_steps=256, sampling='final', img_size=224, return_state=False):
    """
    Rollout a simulation described by the specified substrate and parameters.

    Parameters
    ----------
    rng : jax rng seed for the rollout
    params : parameters to configure the simulation within the substrate
    s0 : (optional) initial state of the simulation. If None, then substrate.init_state(rng, params) is used.
    substrate : the substrate object
    fm : the foundation model object. If None, then no image embedding is calculated.
    rollout_steps : number of timesteps to run simulation for
    sampling : one of either
        - 'final': return only the final state data (default)
        - 'video': return the entire rollout
        - (K, chunk_ends): return the rollout at K sampled intervals
    img_size : image size to render at. Leave at 224 to avoid resizing again for CLIP.
    return_state : return the state data, leave as False, unless you really need it.

    Returns
    ----------
    A dictionary containing
    'state' : the state data of the simulation, None if return_state is False
        shape: (...)
    'rgb' : the image data of the simulation,
        shape (H, W, C)
    'z' : the image embedding of the simulation using the foundation model,
        shape (D)

    If sampling is 'video' then the returned shapes become (rollout_steps, ...).
    If sampling is an int then the returned shapes become (sampling, ...).

    ----------
    This function should be used like this:
    ```
    fm = create_foundation_model('clip')
    substrate = create_substrate('lenia')
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=256, sampling='final', img_size=224, return_state=False)
    rollout_fn = jax.jit(rollout_fn) # jit for speed
    # now you can use rollout_fn as you need...
    rng = jax.random.PRNGKey(0)
    params = substrate.default_params(rng)
    rollout_data = rollout_fn(rng, params)
    ```

    Note: 
    - when sampling is 'final', the function returns data for the T timestep.
    - when sampling is 'video', the function returns data for the [0, ..., T-1] timesteps.
    - when sampling is (K, False), the function returns data for the [0, T//K, T//K * 2, ..., T - T//K] timesteps.
    - when sampling is (K, True), the function returns data for the [T//K, T//K * 2, ..., T] timesteps.
    """

    if s0 is None:
        s0 = substrate.init_state(rng, params)
    embed_img_fn = (lambda img: None) if fm is None else fm.embed_img
    
    if sampling == 'final': # return only the final state
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        state_final, _ = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        img = substrate.render_state(state_final, params=params, img_size=img_size)
        z = embed_img_fn(img)
        return dict(rgb=img, z=z, state=(state_final if return_state else None))
    elif sampling == 'video': # return the entire rollout
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            img = substrate.render_state(state, params=params, img_size=img_size)
            z = embed_img_fn(img)
            return next_state, dict(rgb=img, z=z, state=(state if return_state else None))
        _, data = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        return data
    elif isinstance(sampling, int) or isinstance(sampling, tuple): # return the rollout at K sampled intervals
        K, chunk_ends = sampling if isinstance(sampling, tuple) else (sampling, False)
        assert rollout_steps % K == 0
        chunk_steps = rollout_steps // K
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        def chunk_fn(state, _rng):
            next_state, _ = jax.lax.scan(step_fn, state, split(_rng, chunk_steps))
            state_to_use = next_state if chunk_ends else state
            img = substrate.render_state(state_to_use, params=params, img_size=img_size)
            z = embed_img_fn(img)
            return next_state, dict(rgb=img, z=z, state=(state_to_use if return_state else None))
        state_final, data = jax.lax.scan(chunk_fn, s0, split(rng, K))
        return data
    else:
        raise ValueError(f"sampling {sampling} not recognized")
