"""Runtime friction modification via the PhysX tensor API.

IsaacLab / PhysX only reads the USD stage once at warmup
(``SimulationManager.initialize_physics`` -> ``force_load_physics_from_usd``).
After that, PhysX keeps its own copy of every material, so mutating
``physicsMaterial:staticFriction`` on a USD prim does not change simulated
behavior.

The functions here use the PhysX tensor API
(``root_view.set_material_properties``) to write friction values directly
into PhysX's runtime buffers. These changes take effect on the next simulation
step without any USD reload. IsaacLab's own ``randomize_rigid_body_material``
(``deps/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py``) uses the same
API, but only at init time.

Limitation: the tensor API only exposes static / dynamic friction and
restitution. The combine mode (``max``/``avg``/``min``) is baked into the
PhysX material and cannot be changed at runtime.

Two functions are exposed:
    ``set_rigid_body_friction`` - write friction to whole bodies (= all
        collision shapes within the named links / the entire RigidObject).
    ``set_rigid_body_friction_by_prim_substring`` - write friction to
        only the subset of collision shapes whose USD prim path contains
        a given substring. Used to target the cabinet *handle* shapes
        (paths contain ``handle_col``) without touching the drawer-body
        shapes inside the same link.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.assets import Articulation, RigidObject


def _get_material_properties(view) -> torch.Tensor:
    """Read the view's (num_envs, num_shapes, 3) material buffer as an owned CPU torch tensor.

    Isaac Sim 6's tensor API returns a ``wp.array`` (Isaac Sim 5 returned a torch tensor).
    """
    mats = view.get_material_properties()
    return (wp.to_torch(mats) if isinstance(mats, wp.array) else mats).clone()


def _set_material_properties(view, mats: torch.Tensor, env_ids_cpu: torch.Tensor) -> None:
    """Write a full material buffer back for ``env_ids_cpu`` (Isaac Lab 3 passes warp arrays)."""
    view.set_material_properties(
        wp.from_torch(mats.contiguous(), dtype=wp.float32),
        wp.from_torch(env_ids_cpu.to(torch.int32).contiguous(), dtype=wp.int32),
    )
from isaaclab.managers import SceneEntityCfg


def _get_body_shape_slices(env, asset, asset_name, body_names):
    """Return ``[(body_name, start, end), ...]`` shape-index slices per body.

    Cached on the env as ``_friction_shape_slices_cache`` so the per-body
    physics-view walk only happens once per (asset, body_names) pair.
    """
    cache = getattr(env, "_friction_shape_slices_cache", None)
    if cache is None:
        cache = {}
        env._friction_shape_slices_cache = cache

    key = (asset_name, tuple(body_names))
    if key in cache:
        return cache[key]

    view = asset.root_view
    link_names = list(view.shared_metatype.link_names)

    num_shapes_per_body = []
    for link_path in view.link_paths[0]:
        rb_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        num_shapes_per_body.append(rb_view.max_shapes)

    slices = []
    for name in body_names:
        if name not in link_names:
            raise ValueError(
                f"Body '{name}' not found on asset '{asset_name}'. "
                f"Available link names: {link_names}"
            )
        bid = link_names.index(name)
        start = sum(num_shapes_per_body[:bid])
        end = start + num_shapes_per_body[bid]
        if end == start:
            raise ValueError(
                f"Body '{name}' on asset '{asset_name}' has no collision "
                "shapes; cannot set friction on it."
            )
        slices.append((name, start, end))

    cache[key] = slices
    return slices


def set_rigid_body_friction(
    env,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    body_names: list[str] | None = None,
    static_friction: float = 0.5,
    dynamic_friction: float | None = None,
    verbose: bool = False,
):
    """EventTerm: write PhysX friction on selected bodies at runtime.

    Unlike USD-attribute edits, this change propagates to the physics on the
    very next step because it writes to PhysX's material buffers directly.

    Works for both ``Articulation`` (multi-body - pass ``body_names`` to pick
    which links) and ``RigidObject`` (single-body - ``body_names`` is ignored
    and all collision shapes on the body are updated).

    Args:
        env: ManagerBasedRLEnv instance (supplied by EventManager).
        env_ids: Tensor of env indices, ``slice(None)``, or ``None`` (= all).
        asset_cfg: SceneEntityCfg naming the asset (e.g. ``left_arm`` or ``obj_0``).
        body_names: Link names whose collision shapes will be updated. Required
            for ``Articulation`` assets. Ignored for ``RigidObject`` (which has
            only one body).
        static_friction: New static friction value.
        dynamic_friction: New dynamic friction. Defaults to ``static_friction``.
        verbose: If True, print a ``[FRICTION]`` log line per call.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    view = asset.root_view

    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)[env_ids]
    else:
        env_ids_cpu = env_ids.detach().cpu().to(torch.long)

    if env_ids_cpu.numel() == 0:
        return

    dyn = static_friction if dynamic_friction is None else dynamic_friction

    # get -> modify selected slice -> set back. Sparse set is not supported.
    mats = _get_material_properties(view)

    if isinstance(asset, RigidObject):
        # Single-body rigid object: write to every collision shape.
        # Tensor shape is (num_envs, num_shapes, 3); no per-body slice walk needed.
        num_shapes = mats.shape[1]
        if verbose:
            first_env = int(env_ids_cpu[0].item())
            prev_static = float(mats[first_env, 0, 0].item())
            prev_dynamic = float(mats[first_env, 0, 1].item())
            print(
                f"[FRICTION] asset='{asset_cfg.name}' (rigid) "
                f"shapes=[0:{num_shapes}] envs={env_ids_cpu.tolist()} "
                f"static {prev_static:.3f} -> {static_friction:.3f}, "
                f"dynamic {prev_dynamic:.3f} -> {dyn:.3f}"
            )
        mats[env_ids_cpu, :, 0] = static_friction
        mats[env_ids_cpu, :, 1] = dyn
        _set_material_properties(view, mats, env_ids_cpu)
        return

    # Articulation path - slice per requested body.
    if not body_names:
        raise ValueError(
            f"set_rigid_body_friction: ``body_names`` is required for "
            f"Articulation assets (got asset='{asset_cfg.name}')."
        )
    slices = _get_body_shape_slices(env, asset, asset_cfg.name, body_names)

    if verbose:
        first_env = int(env_ids_cpu[0].item())
        for name, s, e in slices:
            prev_static = float(mats[first_env, s, 0].item())
            prev_dynamic = float(mats[first_env, s, 1].item())
            print(
                f"[FRICTION] asset='{asset_cfg.name}' body='{name}' "
                f"shapes=[{s}:{e}] envs={env_ids_cpu.tolist()} "
                f"static {prev_static:.3f} -> {static_friction:.3f}, "
                f"dynamic {prev_dynamic:.3f} -> {dyn:.3f}"
            )

    for _, s, e in slices:
        mats[env_ids_cpu, s:e, 0] = static_friction
        mats[env_ids_cpu, s:e, 1] = dyn

    _set_material_properties(view, mats, env_ids_cpu)


def _find_shape_indices_by_prim_substring(
    env, asset, asset_name: str, substring: str,
    exclude_substring: str | None = None,
) -> list[int]:
    """Return global PhysX shape indices whose USD prim path contains
    ``substring`` and (optionally) does NOT contain ``exclude_substring``.

    Walks the USD stage once per (asset, substring, exclude_substring)
    and caches the result on the env. Assumes PhysX assigns per-link
    shape indices in USD prim-traversal order (the standard IsaacLab
    behavior for converted URDFs).

    Cabinet collision prims follow the naming convention
    ``{link_name}_col_{idx}`` for body and ``{link_name}_handle_col_{idx}``
    for handle (set in
    ``utils/asset_conversion_utils.py:get_collision_approximation_for_urdf``).
    Pass ``substring='handle_col'`` to target only handle shapes; pass
    ``substring='_col_', exclude_substring='handle_col'`` for the complement.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    cache_attr = f"_shape_indices_cache__{substring}__excl_{exclude_substring or ''}"
    cache = getattr(env, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(env, cache_attr, cache)
    if asset_name in cache:
        return cache[asset_name]

    stage = omni.usd.get_context().get_stage()
    view = asset.root_view
    matched_indices: list[int] = []
    global_offset = 0

    # link_paths[0] = list of link prim paths for env 0. Shape indices
    # for env 0 mirror those for every other env (parallel envs share
    # the same articulation topology).
    for link_path in view.link_paths[0]:
        link_path_str = str(link_path)
        link_prim = stage.GetPrimAtPath(link_path_str)
        if link_prim.IsValid():
            shape_idx_in_link = 0
            for prim in Usd.PrimRange(link_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                prim_path_str = str(prim.GetPath())
                if substring in prim_path_str and (
                    exclude_substring is None or exclude_substring not in prim_path_str
                ):
                    matched_indices.append(global_offset + shape_idx_in_link)
                shape_idx_in_link += 1

        # Advance the global shape offset by this link's shape count.
        rb_view = asset._physics_sim_view.create_rigid_body_view(link_path_str)
        global_offset += rb_view.max_shapes

    cache[asset_name] = matched_indices
    return matched_indices


def set_rigid_body_friction_by_prim_substring(
    env,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    prim_substring: str,
    static_friction: float,
    dynamic_friction: float | None = None,
    exclude_substring: str | None = None,
    verbose: bool = False,
):
    """EventTerm: write PhysX friction to the subset of collision shapes
    whose USD prim paths contain ``prim_substring`` and (optionally) do
    NOT contain ``exclude_substring``.

    Unlike ``set_rigid_body_friction`` which operates at the body/link
    level, this lets you target SPECIFIC collision shapes inside a
    multi-shape link. Use case: an articulation whose link bundles
    geometrically-distinct collision prims (e.g. a drawer link with
    both ``link_*_body_col_*`` shapes and ``link_*_handle_col_*``
    shapes) - passing ``prim_substring='handle_col'`` modulates the
    handle's friction while leaving the drawer body untouched.

    Complementary case - body-only (zero out the drawer face / cabinet
    shell so finger contact there is frictionless), call with
    ``prim_substring='_col_', exclude_substring='handle_col'``.

    Args:
        env, env_ids, asset_cfg: standard event params.
        prim_substring: substring searched in each collision prim's
            USD path. Only shapes whose path contains this are
            modified.
        static_friction: new static friction value for matched shapes.
        dynamic_friction: new dynamic friction. Defaults to
            ``static_friction``.
        exclude_substring: if set, prims whose paths contain this
            substring are skipped even if they match ``prim_substring``.
            Used for "match all collision prims EXCEPT the handle".
        verbose: log a ``[FRICTION]`` line per call.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    view = asset.root_view

    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)[env_ids]
    else:
        env_ids_cpu = env_ids.detach().cpu().to(torch.long)
    if env_ids_cpu.numel() == 0:
        return

    shape_indices = _find_shape_indices_by_prim_substring(
        env, asset, asset_cfg.name, prim_substring, exclude_substring
    )
    if not shape_indices:
        if verbose:
            print(
                f"[FRICTION] asset='{asset_cfg.name}' "
                f"no shapes match prim_substring='{prim_substring}'"
                f"{'' if exclude_substring is None else f', exclude={exclude_substring!r}'}; skipping."
            )
        return

    dyn = static_friction if dynamic_friction is None else dynamic_friction
    mats = _get_material_properties(view)

    if verbose:
        first_env = int(env_ids_cpu[0].item())
        prev_static = float(mats[first_env, shape_indices[0], 0].item())
        prev_dynamic = float(mats[first_env, shape_indices[0], 1].item())
        excl_str = '' if exclude_substring is None else f" exclude='{exclude_substring}'"
        print(
            f"[FRICTION] asset='{asset_cfg.name}' "
            f"prim_substring='{prim_substring}'{excl_str} "
            f"shapes={shape_indices} envs={env_ids_cpu.tolist()} "
            f"static {prev_static:.3f} -> {static_friction:.3f}, "
            f"dynamic {prev_dynamic:.3f} -> {dyn:.3f}"
        )

    for shape_idx in shape_indices:
        mats[env_ids_cpu, shape_idx, 0] = static_friction
        mats[env_ids_cpu, shape_idx, 1] = dyn

    _set_material_properties(view, mats, env_ids_cpu)


def set_articulation_body_mass(
    env,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    mass: float,
    recompute_inertia: bool = True,
    verbose: bool = False,
):
    """EventTerm: write a fixed mass to selected bodies of an articulation.

    Use ``SceneEntityCfg("obj_X", body_names=[...])`` to target specific
    links. Inertia is scaled by the mass ratio so the body's rotational
    dynamics remain proportionally correct (matches IsaacLab's own
    ``randomize_rigid_body_mass`` recompute path).

    Args:
        env, env_ids: standard event params.
        asset_cfg: must resolve to an Articulation; ``body_names`` selects
            which links to override (defaults to ALL links if unset -
            usually NOT what you want for a multi-link articulation).
        mass: new mass value in kg.
        recompute_inertia: if True, scale the body's diagonal inertia by
            the new/old mass ratio.
        verbose: print one ``[MASS]`` line per call.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)[env_ids]
    else:
        env_ids_cpu = env_ids.detach().cpu().to(torch.long)
    if env_ids_cpu.numel() == 0:
        return

    # Resolve body_ids the same way IsaacLab's own mass event does.
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Isaac Lab 3 asset API (mirrors isaaclab.envs.mdp.events.randomize_rigid_body_mass): partial
    # (len(env_ids), len(body_ids)) data through set_masses_index / set_inertias_index.
    env_ids_dev = env_ids_cpu.to(asset.device)
    body_ids_dev = body_ids.to(asset.device)
    masses = asset.data.body_mass.torch.clone()
    prev_masses = masses[env_ids_dev[:, None], body_ids_dev].clone()
    new_masses = torch.full_like(prev_masses, float(mass))
    asset.set_masses_index(masses=new_masses, body_ids=body_ids_dev, env_ids=env_ids_dev)

    if recompute_inertia:
        # Scale inertia by mass ratio (per-body). Avoid div-by-zero by clamping previous mass.
        ratios = (float(mass) / torch.clamp(prev_masses, min=1e-9))
        inertias = asset.data.body_inertia.torch.clone()
        # inertias shape: (num_envs, num_bodies, 9).
        new_inertias = inertias[env_ids_dev[:, None], body_ids_dev] * ratios.unsqueeze(-1)
        asset.set_inertias_index(inertias=new_inertias, body_ids=body_ids_dev, env_ids=env_ids_dev)

    if verbose:
        prev_first = float(prev_masses[0, 0].item()) if prev_masses.numel() > 0 else float("nan")
        print(
            f"[MASS] asset='{asset_cfg.name}' "
            f"body_ids={body_ids.tolist()} envs={env_ids_cpu.tolist()} "
            f"{prev_first:.3f} kg -> {mass:.3f} kg"
        )
