# Attention Stats

============================================================
Scene seed=500
Queries: 32, memory size: 304 = 4 layers x 76 tokens

Per-layer attention mass (averaged over queries):
  layer slot 0: 0.334  #############
  layer slot 1: 0.349  #############
  layer slot 2: 0.113  ####
  layer slot 3: 0.205  ########

Attention entropy per query (max possible = 5.717):
  min: 2.593  mean: 4.355  max: 5.082
  (low entropy = peaky / specific, high = uniform / ignoring memory)

Top-5 attended tokens per query (first 5 queries):
  Q0: L0:t21(<|image_pad|>):0.11  L0:t0(<|vision_start|>):0.05  L0:t72(Ġthe):0.05  L1:t66(Push):0.03  L0:t73(Ġtarget):0.03
  Q1: L0:t29(<|image_pad|>):0.11  L3:t0(<|vision_start|>):0.09  L0:t0(<|vision_start|>):0.06  L0:t72(Ġthe):0.05  L3:t25(<|image_pad|>):0.03
  Q2: L1:t55(<|image_pad|>):0.11  L3:t0(<|vision_start|>):0.06  L1:t42(<|image_pad|>):0.05  L1:t43(<|image_pad|>):0.03  L1:t20(<|image_pad|>):0.03
  Q3: L0:t29(<|image_pad|>):0.12  L0:t21(<|image_pad|>):0.12  L1:t34(<|image_pad|>):0.11  L1:t43(<|image_pad|>):0.10  L1:t47(<|image_pad|>):0.07
  Q4: L3:t10(<|image_pad|>):0.12  L0:t29(<|image_pad|>):0.11  L0:t21(<|image_pad|>):0.07  L1:t28(<|image_pad|>):0.05  L2:t60(<|image_pad|>):0.05

============================================================
Scene seed=1500
Queries: 32, memory size: 304 = 4 layers x 76 tokens

Per-layer attention mass (averaged over queries):
  layer slot 0: 0.298  ###########
  layer slot 1: 0.411  ################
  layer slot 2: 0.110  ####
  layer slot 3: 0.182  #######

Attention entropy per query (max possible = 5.717):
  min: 2.265  mean: 4.216  max: 4.947
  (low entropy = peaky / specific, high = uniform / ignoring memory)

Top-5 attended tokens per query (first 5 queries):
  Q0: L0:t0(<|vision_start|>):0.07  L0:t72(Ġthe):0.07  L3:t0(<|vision_start|>):0.03  L0:t73(Ġtarget):0.03  L1:t75(.):0.03
  Q1: L3:t0(<|vision_start|>):0.13  L0:t0(<|vision_start|>):0.09  L0:t72(Ġthe):0.09  L1:t47(<|image_pad|>):0.02  L1:t73(Ġtarget):0.02
  Q2: L1:t47(<|image_pad|>):0.12  L1:t53(<|image_pad|>):0.12  L3:t0(<|vision_start|>):0.05  L1:t42(<|image_pad|>):0.03  L0:t0(<|vision_start|>):0.03
  Q3: L0:t22(<|image_pad|>):0.25  L1:t11(<|image_pad|>):0.12  L1:t35(<|image_pad|>):0.10  L1:t27(<|image_pad|>):0.06  L1:t46(<|image_pad|>):0.04
  Q4: L3:t10(<|image_pad|>):0.16  L1:t47(<|image_pad|>):0.13  L2:t63(<|image_pad|>):0.06  L0:t0(<|vision_start|>):0.04  L0:t72(Ġthe):0.04

============================================================
Scene seed=2500
Queries: 32, memory size: 304 = 4 layers x 76 tokens

Per-layer attention mass (averaged over queries):
  layer slot 0: 0.340  #############
  layer slot 1: 0.396  ###############
  layer slot 2: 0.108  ####
  layer slot 3: 0.156  ######

Attention entropy per query (max possible = 5.717):
  min: 2.143  mean: 4.218  max: 5.023
  (low entropy = peaky / specific, high = uniform / ignoring memory)

Top-5 attended tokens per query (first 5 queries):
  Q0: L0:t52(<|image_pad|>):0.10  L0:t0(<|vision_start|>):0.06  L0:t72(Ġthe):0.05  L0:t73(Ġtarget):0.03  L1:t75(.):0.03
  Q1: L3:t0(<|vision_start|>):0.12  L0:t0(<|vision_start|>):0.07  L0:t72(Ġthe):0.06  L0:t22(<|image_pad|>):0.06  L1:t75(.):0.02
  Q2: L1:t14(<|image_pad|>):0.13  L1:t5(<|image_pad|>):0.09  L1:t45(<|image_pad|>):0.04  L3:t0(<|vision_start|>):0.04  L1:t42(<|image_pad|>):0.04
  Q3: L0:t22(<|image_pad|>):0.13  L0:t52(<|image_pad|>):0.13  L1:t42(<|image_pad|>):0.11  L1:t4(<|image_pad|>):0.10  L1:t14(<|image_pad|>):0.09
  Q4: L1:t14(<|image_pad|>):0.25  L2:t38(<|image_pad|>):0.10  L3:t18(<|image_pad|>):0.10  L0:t4(<|image_pad|>):0.08  L1:t38(<|image_pad|>):0.04

============================================================
Cross-scene attention diff (Q-Pooler scene-specificity)
============================================================
  scene0 vs scene1: mean_abs_diff=0.0029  relative=0.880
  scene0 vs scene2: mean_abs_diff=0.0030  relative=0.926
  scene1 vs scene2: mean_abs_diff=0.0029  relative=0.882
  (higher = attention is more scene-specific)
