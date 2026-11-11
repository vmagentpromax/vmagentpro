import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vmagent.gpu.environment import NodeSpec,TaskSpec,GpuSchedEnv
from harness import ExperimentEnv,evaluate,CORE_POLICIES

def task(pid,gpu,arrival,service,priority):
    return TaskSpec(pid,gpu,gpu,('H20',),arrival,service,priority)

class Checks(unittest.TestCase):
    def env(self,tasks,variant='paper_cost',window=(0,100)):
        return ExperimentEnv([NodeSpec('n','c','H20',2,8)],tasks,variant=variant,
             clock='fast',checkpoint_interval=10,measurement_window=window,max_sim_time=1000)

    def test_lost_progress_and_first_response(self):
        env=self.env([task('lp',2,0,50,'LP'),task('hp',2,13,5,'HP')])
        env.run_experiment();m,rows=evaluate(env);lp=next(r for r in rows if r['pod_id']=='lp')
        self.assertEqual(lp['first_start'],0);self.assertEqual(lp['last_start'],18)
        self.assertEqual(lp['wasted_sec'],3);self.assertEqual(lp['running_sec'],53)
        self.assertEqual(lp['total_queue_sec'],5);self.assertEqual(lp['finish'],58)
        self.assertAlmostEqual(m['wasted_gpu_hours_full_run'],6/3600)

    def test_checkpoint_boundary_no_loss(self):
        env=self.env([task('lp',2,0,50,'LP'),task('hp',2,10,5,'HP')])
        env.run_experiment();m,_=evaluate(env)
        self.assertEqual(m['wasted_gpu_hours_full_run'],0)

    def test_random_deterministic(self):
        ts=[task('a',1,0,50,'LP'),task('b',1,0,50,'LP'),task('hp',1,13,5,'HP')]
        x=self.env(ts,'random_victim');y=self.env(ts,'random_victim')
        x.run_experiment();y.run_experiment();self.assertEqual(x.events,y.events)

    def test_no_preempt_preserves_hp_wait(self):
        env=self.env([task('lp',2,0,50,'LP'),task('hp',2,13,5,'HP')],'no_preempt')
        env.run_experiment();m,rows=evaluate(env)
        self.assertEqual(m['preemption_victims_window'],0)
        self.assertEqual(next(r for r in rows if r['pod_id']=='hp')['response_sec'],37)

    def test_loss_clipped_across_window(self):
        env=self.env([task('lp',2,0,50,'LP'),task('hp',2,13,5,'HP')],window=(11,12))
        env.run_experiment();m,_=evaluate(env)
        self.assertEqual(m['wasted_gpu_ratio_window'],1)
        self.assertEqual(m['useful_gpu_ratio_retrospective'],0)

    def test_hp_reclaims_before_new_lp(self):
        ts=[task('a',1,0,100,'LP'),task('b',1,0,10,'LP'),
            task('new_lp',1,10,100,'LP'),task('hp',2,10,5,'HP')]
        fixed=self.env(ts);legacy=self.env(ts,'legacy_spotgpu')
        fixed.run_experiment();legacy.run_experiment()
        self.assertEqual(fixed.records['new_lp']['preemption_count'],0)
        self.assertGreater(legacy.records['new_lp']['preemption_count'],0)

    def test_original_placement_variants_match_core(self):
        ts=[task('a',1,0,10,'LP'),task('b',1,0,20,'HP'),task('c',2,1,5,'HP')]
        nodes=[NodeSpec('n1','c','H20',2,8),NodeSpec('n2','c','H20',2,8)]
        for policy in CORE_POLICIES:
            with self.subTest(policy=policy):
                original=GpuSchedEnv(nodes,ts,clock='fast',measurement_window=(0,100),
                                    max_sim_time=1000,seed=42)
                experiment=ExperimentEnv(nodes,ts,variant=policy,clock='fast',
                                         measurement_window=(0,100),max_sim_time=1000,seed=42)
                original.run(policy)
                experiment.run_experiment()
                self.assertEqual(original.events,experiment.events)
                self.assertEqual(original.records,experiment.records)
                self.assertEqual(experiment.summary()['preemptions'],0)

    def test_legacy_spotgpu_matches_core(self):
        ts=[task('lp',2,0,50,'LP'),task('hp',2,13,5,'HP')]
        nodes=[NodeSpec('n','c','H20',2,8)]
        original=GpuSchedEnv(nodes,ts,clock='fast',measurement_window=(0,100),
                             max_sim_time=1000,checkpoint_interval=10,seed=42)
        experiment=ExperimentEnv(nodes,ts,variant='legacy_spotgpu',clock='fast',
                                 measurement_window=(0,100),max_sim_time=1000,
                                 checkpoint_interval=10,seed=42)
        original.run('spotgpu')
        experiment.run_experiment()
        self.assertEqual(original.events,experiment.events)
        self.assertEqual(original.records,experiment.records)

if __name__=='__main__':unittest.main()
