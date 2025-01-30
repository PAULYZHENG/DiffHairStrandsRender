This repo contains the hair strands render in HairStep, which is modified from [SoftRas](https://github.com/ShichenLiu/SoftRas).

Please install the conda environment of [HairStep](https://github.com/GAP-LAB-CUHK-SZ/HairStep) first.

Then run:
  ```
python setup_hair.py install

mv ./build/lib.linux-x86_64-3.6 ./build/lib

python -m app.render_orien_batch
python -m app.render_depth_batch
  ```

Note that the camera parameters are from PIFU. 