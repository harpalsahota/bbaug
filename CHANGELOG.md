# 0.4.2
- Fixed bug where bounding box specific augmentations would not be clipped or removed

# 0.4.1
- Now possible to pass a random state to the policy container ensuring reproducible augmentations

# 0.4.0
- Apply augmentations to bounding boxes individually
- Fixed bug in `visualise_policy`

# 0.3.0
- Class labels are now required for bounding boxes

# 0.2.1
- Fixed bug where the cutout would be larger than the image

# 0.2.0
- Implementation of policy version 0, 1 and 2
- Module to aid in the visualisation of a policy
- Notebooks for bbaug integration into training model and custom policies

# 0.1.0
- Implementation of policy version 3