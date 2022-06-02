1. 
```
 conda create --name electra python=3.7
 conda activate electra
```
2. 
```
conda install cudatoolkit=10.0 cudnn=7.3.1
pip3 install tensorflow-gpu==1.15
# Verify install:
python3 -c "import tensorflow as tf"
```

**After importing tensorflow error occur:**
```
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
```

Downgrade protobuf:
```
pip install --upgrade "protobuf<=3.20.1"
```

then everything is fine:
```
python3 -c "import tensorflow as tf"
```


3. Copy OpenWebTextCorpus and extract to `./datasets/` from  [here](https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx).

4. Get vocab.txt

```
wget  https://storage.googleapis.com/electra-data/vocab.txt -O ./datasets/vocab.txt 
```


5. Run preprocessing of dataset: 
``` 
python3 build_openwebtext_pretraining_dataset.py --data-dir ./datasets --num-processes 20 
```

6. Open interactive session: 
` srun --qos=8gpu7d --time 3-0  --gres=gpu:1 --partition=common --constraint=homedir  --pty /bin/bash -i`
 (`qos `should be set according to your privileges) or run training directly from `srun`.

7. Once again activate environment: 
```
conda activate electra
```

8. Train:
```
 python3 run_pretraining.py --data-dir ./datasets --model-name electra_small_owt
```