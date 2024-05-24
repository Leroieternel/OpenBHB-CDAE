For running vanilla unet (main_0322.py):
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/ae.yaml

For running dae unet (main_dae.py):
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/dae.yaml

For running vanilla unet (main_0322.py):
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/cdae_age_mlp.yaml

For evaluating age and site scores:
  cd ../contrastive-brain-age-prediction/src
  python age_site_score.py
