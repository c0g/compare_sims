import mujoco as mj
import numpy as np
from mujoco import mjx
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
import mujoco
from PIL import Image
import time
import os


from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import queue
import tqdm
import io
import threading

class JitStep:
    def __init__(self, mjx_model):
        self.model = mjx_model
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))        

    def __call__(self, rngs, batch):
        return rngs, self.jit_step(self.model, batch)

def random_init(rng, batch):
    rng, _rng = jax.random.split(rng, 2)
    batch = batch.replace(qpos=jax.random.uniform(_rng, batch.qpos.shape))
    return rng, batch

def brownian_control(rng, batch):
    rng, _rng = jax.random.split(rng, 2)
    d_ctrl = jax.random.normal(_rng, batch.ctrl.shape)
    batch = batch.replace(ctrl=batch.ctrl + d_ctrl)
    return rng, batch


class Simulator():
    def __init__(self, xml_path, sim_duration: int, n_parallel: int, seed=1337):
        xml = open(xml_path).read()
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        self.duration = sim_duration
        self.key = key = jax.random.PRNGKey(seed)
        self.n_parallel = n_parallel

    def simulate(self):
        self.key, rng = jax.random.split(self.key, 2)
        batch = jax.vmap(lambda _: self.mjx_data)(range(self.n_parallel))
        rng, batch = random_init(rng, batch)

        sim = JitStep(self.mjx_model)
        # run once outside of tqdm to compile
        brownian_control(rng, batch)
        rng, batch = sim(rng, batch)
        with tqdm.tqdm(total=self.duration) as bar:
            while True:
                brownian_control(rng, batch)
                rng, batch = sim(rng, batch)
                bar.update(1)

n = 64*64
oleg = Simulator('../models/oleg.xml', sim_duration=1.0, n_parallel=n)
oleg.simulate()