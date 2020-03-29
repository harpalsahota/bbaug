![Alt text](./coverage.svg) ![Develop Branch CI](https://github.com/harpalsahota/bbaug/workflows/Develop%20Branch%20CI/badge.svg?branch=develop)


# BBAug

BBAug is a Python package for the implementations of Google’s Brain Team’s bounding box augmentations policies. 
The package is aimed for PyTorch users who wish to use these policies in the augmentation of bounding boxes during the 
training of a model.

**References**
- [Paper](https://arxiv.org/abs/1906.11172)
- [Tensorflow Policy Code](https://github.com/tensorflow/tpu/blob/2264f53d95852efbfb82ea27f03ca749e1205968/models/official/detection/utils/autoaugment_utils.py)

## Features

- [x] Implementation of version 3 of policies
- [x] Custom policies
- [x] Custom augmentations
- [x] Bounding boxes are removed if they fall outside of the image
- [x] Boudning boxes are clipped if they are partially outside the image
- [x] Augmentations that imply direction e.g. rotation is randomly determined

## To Do
- [ ] Implementation of version 2 of policies
- [ ] Implementation of version 1 of policies
- [ ] For bounding box augmentations apply the probability individually for each box not collectively

## Installation

Installation is best done via pip:
> pip install bbaug

### Prerequisites
- Python 3.6+
- PyTorch
- Torchvision

## Comparisons to Tensorflow Implementation

## Description and Usage

For detailed description on usage please refer to the Python notebooks provided in the `notebooks` folder.

A augmentation is define by 3 attributes:
- **Name**: Name of the augmentation
- **Probability**: Probability of augmentation being applied
- **Magnitude**: The degree of the augmentation (values are integers between 0 and 10)

A policy is a collection of augmentations: e.g.
```python
single_policy = [('translation', 0.5, 1), ('rotation', 1.0, 9)]
```
In the above example we have two augmentations in a single policy. The `translation` policy has a 
probability of 0.5 and a magnitude of 1, whereas the `rotation` policy has a probability of 1.0 and a 
magnitude of 9. The magnitudes do not directly translate into the augmentation policy i.e. a magnitude of 9
does not mean a 9 degree rotation. Instead, scaling is applied to the magnitude to determine the value passed
to the augmentation method. The scaling varies depending on the augmentation used.

A policy is a set of single policies:
```python
policies = [
    [('translation', 0.5, 1), ('rotation', 1.0, 9)],
    [('colour', 0.5, 1), ('cutout', 1.0, 9)],
    [('rotation', 0.5, 1), ('solarize', 1.0, 9)]
]
``` 
During training a random policy is selected from the list of policies and applied to the image and because
each augmentation has it's own probability this adds a degree of stochasticity to training. 

### Policies

Each augmentation contains a string referring to the name of the augmentation. The `augmentations` module
contains a dictionary mapping the name to a method reference of the augmentation.
```python
from bbaug.augmentations import NAME_TO_AUGMENTATION
print(NAME_TO_AUGMENTATION) # Shows the dictionary of the augmentation name to the method reference
```

#### Listing All Policies Available
To obtain a list of all available polices run the `list_policies` method. This will return a list of strings
containing the function names for the policy sets.
```python
from bbaug.policies import list_policies
print(list_policies()) # List of policies available
```
 
#### Listing the policies in a policy set
```python
from bbaug.policies import policies_v3
print(policies_v3()) # Will list all the polices in version 3
```

#### Visualising a Policy
 TODO

#### Policy Container
To help integrate the policies into trained there is a `PolicyContainer` class available in the `policies`
module. The container accepts the following inputs:
- **policy_set** (required): The policy set to use
- **name_to_augmentation** (optional, default: augmentations.NAME_TO_AUGMENTATION): The dictionary mapping the augmentation name to the method referece
- **return_yolo** (optional, default: False): Return the bounding boxes in YOLO format otherwise `[x_min, y_min, x_man, y_max]` in pixels is returned 

Usage of the policy container:
```python
from bbaug import policies

# select policy v3 set
aug_policy = policies.policies_v3()
 
# instantiate the policy container with the selected policy set
policy_container = policies.PolicyContainer(aug_policy)

# select a random policy from the policy set
random_policy = policy_container.select_random_policy() 

# Apply the augmentation. Returns the augmented image and bounding boxes.
# Image is a numpy array of the image
# Bounding boxes is a list of list of bounding boxes in pixels.
# e.g. [[x_min, y_min, x_man, y_max], [x_min, y_min, x_man, y_max]]
img_aug, bbs_aug = policy_container.apply_augmentation(random_policy, image, bounding_boxes)
```