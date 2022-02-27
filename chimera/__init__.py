from gym.envs.registration import register

register(
    # Format should be xxx-v0, xxx-v1....
    id='Prim-v0',
    # Expalined in envs/__init__.py
    entry_point='chimera.envs:PrimEnv',
)

register(
    id='Mesh-v0',
    entry_point='chimera.envs:MeshEnv',
)