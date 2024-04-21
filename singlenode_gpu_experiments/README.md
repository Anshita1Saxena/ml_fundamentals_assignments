# Code for Tensorflow GPU training on E1 device

This folder contains the code for tensorflow mnist training.

E1 device is equipped with AMD Rayzen APU, Rocm official versions are not supported. Hence we need to Install unofficial versions of Rocm. Refer to **Device GPU capability** section in the document [Distributed training on E1 devices](docs/distributed_training_on_E1_devices.md) for requirements and installation steps.

```python
python mnist_tf.py
```

