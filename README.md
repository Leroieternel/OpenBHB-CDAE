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

For running cdae without skip connection (main_cdae_age_mlp_wo_sc.py)
  cd ../contrastive-brain-age-prediction/src
  python launcher.py exp/cdae_age_mlp_wo_sc.yaml

U-Net models:
1. (64 classes) normal U-Net encoder/decoder parts: unet_encoder.py  unet_decoder.py
2. (64 classes) without skip connection: net_encoder_wo_sc.py unet_decoder_wo_sc.py
3. (5 classes)  normal U-Net encoder/decoder parts: unet_encoder_balanced.py unet_decoder_balanced.py
4. (5 classes)  without skip connection: unet_en_wo_sc.py   unet_de_wo_sc.py
  
