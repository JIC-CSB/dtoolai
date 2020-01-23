import numpy as np

from dtoolai.data import create_tensor_dataset_from_arrays

output_base_uri = 'tests/data'
output_name = 'example_tensor_dataset'

data_array = np.zeros((100, 81), dtype=np.float32)
label_array = np.zeros(100, dtype=np.uint8)

readme_content = """---

This is an example TensorDataSet.
"""

create_tensor_dataset_from_arrays(
    output_base_uri,
    output_name,
    data_array,
    label_array,
    (1, 9, 9),
    readme_content
)
