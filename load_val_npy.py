import numpy as np

data = np.load('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/test.npy')
print(data.shape)

internal_val = data[: 362, :]
external_val = data[362:, :]
print(internal_val.shape)
print(external_val.shape)
np.save('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/internal_val.npy', internal_val)
np.save('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/external_val.npy', external_val)