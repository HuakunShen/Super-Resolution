# Super Resolution Models

All models are in this directory.

To use the models

```python
# sample code
from model.FSRCNN import FSRCNN
from model.SRCNN import SRCNN
from model.UNetSR import UNetSR, UNetD4, UNetNoTop
```

To initialise the models

```python
'FSRCNN': FSRCNN(factor=3)
'FSRCNN_Original': FSRCNN_Original(scale_factor=3, num_channels=3)
'SRCNN': SRCNN(in_channel=3)
'UNetSR': UNetSR(in_c=3, out_c=3)
'UNetD4': UNetD4()
'UNetNoTop': UNetNoTop()
```

For more details, check each python and their docstring.

## Special Cases

`UNetSR`: 

- for 300x300 input, pass a parameter `output_paddings=[1, 1]` to the constructor
- for 600x600 input, pass a parameter `output_paddings=[1, 0]` to the constructor



































