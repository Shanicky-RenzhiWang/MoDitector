import os

repair_entry_point = 'ads.systems.RLRepair.Roach:ROACHRepairAgent'
ckpt_path = '/home/erdos/workspace/results/MoDitector/debug_rl/rl_repair/ckpt/ckpt_last.pth'

if repair_entry_point is not None and ckpt_path is not None and os.path.isdir(ckpt_path):
    self._repair_policy_class = load_entry_point(repair_entry_point)
    self._repair_ckpt = ckpt_path
    self._repair_policy, _ = self._repair_policy_class.load(self._repair_ckpt)
    self._repair_policy = self._repair_policy.eval()
    logger.info('Loaded Repair Policy')
else:
    self._repair_policy = None
