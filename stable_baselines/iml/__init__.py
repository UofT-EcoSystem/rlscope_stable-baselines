#
# IML: support for "irregular" benchmarking toolkit annotations.
#
import iml_profiler.api as iml

from iml_profiler.parser.common import CATEGORY_SIMULATOR_CPP
from iml_profiler.profiler.clib_wrap import CFuncWrapper

import pybullet
import pybullet_envs
import pybullet_utils

import inspect
import functools

def wrap_pybullet():

    iml.wrap_entire_module(
        'pybullet',
        category=CATEGORY_SIMULATOR_CPP)
    _wrap_bullet_clients()

    # import pybullet
    # iml.wrap_module(
    #     pybullet, category=CATEGORY_SIMULATOR_CPP,
    #     ignore_func_regex=ignore_func_regex,
    #     debug=True)

def unwrap_pybullet():

    _unwrap_bullet_clients()
    iml.unwrap_entire_module('pybullet')

    # import pybullet
    # iml.unwrap_module(pybullet)

#
# IML: pybullet specific work-around!
#
# pybullet python library does some weird dynamic stuff when accessing shared-library functions.
# Basically BulletClient class is checking whether a function that's getting fetched is a built-in.
# If it is, then an extra physicsClientId argument is being given to it.
# So, when we manually wrap this library, the inspect.isbuiltin check will FAIL, and physicsClientId WON'T get supplied!
# So, to work around this, we must also wrap the BulletClient class, and forward physicsClientId.

OldBulletClients = dict()
# pybullet has 3 different implementations of the BulletClient class that essentially look the same.
import pybullet_envs.bullet.bullet_client
import pybullet_envs.minitaur.envs.bullet_client
import pybullet_utils.bullet_client
def _wrap_bullet_clients():
    OldBulletClients['pybullet_envs.bullet.bullet_client.BulletClient'] = pybullet_envs.bullet.bullet_client.BulletClient
    OldBulletClients['pybullet_envs.minitaur.envs.bullet_client.BulletClient'] = pybullet_envs.minitaur.envs.bullet_client.BulletClient
    OldBulletClients['pybullet_utils.bullet_client.BulletClient'] = pybullet_utils.bullet_client.BulletClient
    pybullet_envs.bullet.bullet_client.BulletClient = MyBulletClient
    pybullet_envs.minitaur.envs.bullet_client.BulletClient = MyBulletClient
    pybullet_utils.bullet_client.BulletClient = MyBulletClient
def _unwrap_bullet_clients():
    pybullet_envs.bullet.bullet_client.BulletClient = OldBulletClients['pybullet_envs.bullet.bullet_client.BulletClient']
    pybullet_envs.minitaur.envs.bullet_client.BulletClient = OldBulletClients['pybullet_envs.minitaur.envs.bullet_client.BulletClient']
    pybullet_utils.bullet_client.BulletClient = OldBulletClients['pybullet_utils.bullet_client.BulletClient']

class MyBulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        if (self._client < 0):
            # print("options=", options)
            self._client = pybullet.connect(connection_mode, options=options)
        self._shapes = {}

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if (
                inspect.isbuiltin(attribute) or
                ( isinstance(CFuncWrapper) and inspect.isbuiltin(attribute.func) )
        ) and name not in [
            "invertTransform",
            "multiplyTransforms",
            "getMatrixFromQuaternion",
            "getEulerFromQuaternion",
            "computeViewMatrixFromYawPitchRoll",
            "computeProjectionMatrixFOV",
            "getQuaternionFromEuler",
        ]:  # A temporary hack for now.
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute
