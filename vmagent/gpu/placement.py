"""Placement policies: rank the nodes that a pending task already fits on.

Every policy here is a pure ranking over `_fits`-valid candidates, so a new policy
is one function plus one name in POLICIES — it cannot change what is feasible, only
which feasible node is chosen. That boundary is where M20 (FGD-style fragmentation
aware placement) and M24 (entropy / topology aware) will attach.

`spotgpu` is the exception: its *placement* ranking lives here, but its preemption
half lives in preemption.py, and the two are wired together only by
`GpuSchedEnv.run("spotgpu")` setting `preempt_enabled`. Keeping them apart matters
when reading results: the paper's three placement criteria and the HP-evicts-LP rule
are separate mechanisms, and the repo needs to be able to run one without the other.
"""


class PlacementMixin:
    def choose_action(self, policy="first_fit"):
        obs = self.observation()
        valid = [i for i, allowed in enumerate(obs["avail"]) if allowed]
        if not valid:
            raise RuntimeError("No placement decision is ready")
        if policy == "first_fit":
            return valid[0]
        if policy == "random":
            return self.rng.choice(valid)
        if policy == "best_fit":
            task = self.by_id[self.pending_id]
            return min(valid, key=lambda j: ((self.free_gpu[j]-task.gpu)/self.nodes[j].gpu_capacity
                       +(self.free_cpu[j]-task.cpu)/self.nodes[j].cpu_capacity, j))
        if policy == "worst_fit":
            task = self.by_id[self.pending_id]
            return min(valid, key=lambda j: (-((self.free_gpu[j]-task.gpu)/self.nodes[j].gpu_capacity
                       +(self.free_cpu[j]-task.cpu)/self.nodes[j].cpu_capacity), j))
        raise ValueError("Unknown placement policy: " + str(policy))

    def _spotgpu_choose_action(self):
        """Non-preemptive placement ranking: task packing, homogeneous
        co-location, then eviction history (SpotGPU paper criteria 1-3).

        The terms are applied as a prefix, so `placement_terms=("packing",)` is the
        paper's own ablation baseline and `("packing","co_locate")` removes only the
        eviction-history preference.
        """
        task = self.by_id[self.pending_id]
        priority = self.task_priority.get(self.pending_id, "Other")
        valid = [j for j in range(len(self.nodes)) if self._fits(task, j)]
        if not valid:
            raise RuntimeError("SpotGPU: pending task does not fit any node")

        def packing(j):
            # The historical term is an absolute GPU count, and it stays that way when
            # CPU is not consulted: normalising it would be a silent ordering change on
            # any pool whose nodes differ in GPU capacity. Normalisation only appears
            # once CPU enters the score, because a GPU count and a core count cannot be
            # added without putting them on a common scale first.
            leftover_gpu = self.free_gpu[j]-task.gpu
            if self.placement_cpu_aware == "none":
                return [leftover_gpu]
            scaled = leftover_gpu/self.nodes[j].gpu_capacity
            leftover_cpu = (self.free_cpu[j]-task.cpu)/self.nodes[j].cpu_capacity
            if self.placement_cpu_aware == "lexicographic":
                # GPU leftover decides first; CPU only breaks a GPU tie. This is the
                # defragmentation benchmark's best_fit, transplanted here so the two
                # definitions can be compared inside one policy.
                return [scaled, leftover_cpu]
            return [scaled + leftover_cpu]

        def rank(j):
            key = packing(j) if "packing" in self.placement_terms else [0.]
            if "co_locate" in self.placement_terms:
                resident = self.node_hp_running if priority == "HP" else self.node_lp_running
                key.append(0 if resident[j] > 0 else 1)
            if "history" in self.placement_terms:
                key.append(-self.node_evictions[j] if priority == "HP" else self.node_evictions[j])
            return tuple(key) + (j,)
        return min(valid, key=rank)
