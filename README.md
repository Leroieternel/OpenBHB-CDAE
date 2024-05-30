For running vanilla unet (main_0322.py):
```bash

  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/ae.yaml

For running dae unet (main_dae.py):
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/dae.yaml

For running cdae unet (main_cdae_age_mlp.py):
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/cdae_age_mlp.yaml

For evaluating age and site scores:
  cd ../contrastive-brain-age-prediction/src
  python age_site_score.py

For running cdae without skip connection (main
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/cdae_age_mlp_wo_sc.yaml
  
