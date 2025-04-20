import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import time
import numpy as np

def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# T1
dy_dx1 = grad(f, argnums=0)
dy_dx2 = grad(f, argnums=1)

# Test values
x1_val = 2.0
x2_val = 5.0

f_val = f(x1_val, x2_val)
grad_x1_val = dy_dx1(x1_val, x2_val)
grad_x2_val = dy_dx2(x1_val, x2_val)

print(f"f({x1_val}, {x2_val}) = {f_val}")
print(f"df/dx1 at ({x1_val}, {x2_val}) = {grad_x1_val}")
print(f"df/dx2 at ({x1_val}, {x2_val}) = {grad_x2_val}")

# T2
print("Jaxpr for f:")
print(jax.make_jaxpr(f)(x1_val, x2_val))

print("Jaxpr for df/dx1:")
jaxpr_dy_dx1 = jax.make_jaxpr(dy_dx1)(x1_val, x2_val)
print(jaxpr_dy_dx1)

print("Jaxpr for df/dx2:")
jaxpr_dy_dx2 = jax.make_jaxpr(dy_dx2)(x1_val, x2_val)
print(jaxpr_dy_dx2)

# T3
x1_val = 2.0
x2_val = 5.0

def trace_df_dx1(a, b):
    print("df/dx1:")
    print(f"Input: a={a}, b={b}")
    
    c = jnp.log(a)
    print(f"c = log(a) = {c}")
    
    d = a * b
    print(f"d = a * b = {d}")
    
    e = c + d
    print(f"e = c + d = {e}")
    
    f_val = jnp.sin(b)
    print(f"f = sin(b) = {f_val}")
    
    result = e - f_val
    print(f"_ = e - f = {result}")
    
    g = 1.0 * b
    print(f"g = 1.0 * b = {g}")
    
    h = 1.0 / a
    print(f"h = 1.0 / a = {h}")
    
    i = g + h
    print(f"i = g + h = {i}")
    
    print(f"df/dx1 = {i}")
    return i

def trace_df_dx2(a, b):
    print("df/dx2:")
    print(f"Input: a={a}, b={b}")
    
    c = jnp.log(a)
    print(f"c = log(a) = {c}")
    
    d = a * b
    print(f"d = a * b = {d}")
    
    e = c + d
    print(f"e = c + d = {e}")
    
    f_val = jnp.sin(b)
    print(f"f = sin(b) = {f_val}")
    
    g = jnp.cos(b)
    print(f"g = cos(b) = {g}")
    
    result = e - f_val
    print(f"_ = e - f = {result}")
    
    h = -1.0
    print(f"h = -1.0 = {h}")
    
    i = h * g
    print(f"i = h * g = {i}")
    
    j = a * 1.0
    print(f"j = a * 1.0 = {j}")
    
    k = i + j
    print(f"k = i + j = {k}")
    
    print(f"df/dx2 = {k}")
    return k

traced_dx1 = trace_df_dx1(x1_val, x2_val)
traced_dx2 = trace_df_dx2(x1_val, x2_val)

# T4
print("\nHLO for df/dx1:")

hlo_dy_dx1 = jax.jit(dy_dx1).lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text()
print(hlo_dy_dx1)

print("\nHLO for df/dx2:")
hlo_dy_dx2 = jax.jit(dy_dx2).lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text()
print(hlo_dy_dx2)

# T5
g1 = lambda x1, x2: (jax.jit(f)(x1, x2), jax.jit(dy_dx1)(x1, x2), jax.jit(dy_dx2)(x1, x2))
g2 = jax.jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)))


import timeit
g1_time_cpu = timeit.timeit(lambda: g1(x1_val, x2_val), number=1000) / 1000
g2_time_cpu = timeit.timeit(lambda: g2(x1_val, x2_val), number=1000) / 1000

print(f"\nCPU timings:")
print(f"g1 on CPU: {g1_time_cpu * 1e6:.2f} ms")
print(f"g2 on CPU: {g2_time_cpu * 1e6:.2f} ms")

# Time g1 and g2 on GPU if available
if jax.devices("gpu"):
    x1_val_gpu = jax.device_put(x1_val, jax.devices("gpu")[0])
    x2_val_gpu = jax.device_put(x2_val, jax.devices("gpu")[0])
    
    g1_time_gpu = timeit.timeit(lambda: g1(x1_val_gpu, x2_val_gpu), number=1000) / 1000
    g2_time_gpu = timeit.timeit(lambda: g2(x1_val_gpu, x2_val_gpu), number=1000) / 1000
    
    print(f"\nGPU timings:")
    print(f"g1 on GPU: {g1_time_gpu * 1e6:.2f} ms")
    print(f"g2 on GPU: {g2_time_gpu * 1e6:.2f} ms")

hlo_g1 = jax.jit(g1).lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text()
print("HLO for g1 (individual):")
print(hlo_g1)
hlo_g2 = jax.jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2))).lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text()
print("HLO for g2 (combined):")
print(hlo_g2)

# T6
x1s = jnp.linspace(1.0, 10.0, 1000)
x2s = x1s + 1.0

# Case (a): Vectorization across both parameters
g2_vmap_both = jax.vmap(g2, in_axes=(0, 0))
result_both = g2_vmap_both(x1s, x2s)

# Case (b): Vectorization across the first parameter
g2_vmap_first = jax.vmap(g2, in_axes=(0, None))
result_first = g2_vmap_first(x1s, 0.5)

print("\nboth parameters:")
jaxpr_vmap_both = jax.make_jaxpr(g2_vmap_both)(x1s, x2s)
print(jaxpr_vmap_both)

print("\nfirst parameter:")
jaxpr_vmap_first = jax.make_jaxpr(g2_vmap_first)(x1s, 0.5)
print(jaxpr_vmap_first)