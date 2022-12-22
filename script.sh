export PYTORCH_ENABLE_MPS_FALLBACK=1

python sea_thru.py --original Data/3585_3685/Raw/T_S03647.ARW  --mode Map --size 512 --hint Data/3585_3685/depthMaps/depthT_S03647.tif --depth Data/3585_3685/depthMaps/depthT_S03647.tif --prefix 3647_map_
python sea_thru.py --original Data/3585_3685/Raw/T_S03647.ARW  --mode Hybrid --size 512 --hint Data/3585_3685/depthMaps/depthT_S03647.tif --depth Data/3585_3685/depthMaps/depthT_S03647.tif --prefix 3647_hybrid_
python sea_thru.py --original Data/3585_3685/Raw/T_S03647.ARW  --mode Predict --size 512 --hint Data/3585_3685/depthMaps/depthT_S03647.tif --depth Data/3585_3685/depthMaps/depthT_S03647.tif --prefix 3647_predict_

python graph.py --original Data/D3/Raw/T_S04910.ARW --mode Map --depth Data/D3/depthMaps/depthT_S04910.tif --prefix 

python graph.py --original Data/3585_3685/Raw/T_S03647.ARW --mode Hybrid --depth Data/3585_3685/depthMaps/depthT_S03647.tif --prefix 3647_

python sea_thru.py --original Data/3585_3685/Raw/T_S03585.ARW  --mode Map --size 512 --hint Data/3585_3685/depthMaps/depthT_S03585.tif --depth Data/3585_3685/depthMaps/depthT_S03585.tif --prefix 3585_map_

python sea_thru.py --original Data/D5/Raw/LFT_3377.NEF  --mode Hybrid --size 2048 --hint Data/D5/depthMaps/depthLFT_3377.tif --depth Data/D5/depthMaps/depthLFT_3377.tif --prefix 3377_hybrid

python graph.py --original Data/D5/Raw/LFT_3377.NEF --mode Hybrid --depth Data/D5/depthMaps/depthLFT_3377.tif --prefix 3377_

python sea_thru.py --original Data/D5/Raw/LFT_3379.NEF  --mode Hybrid --size 2048 --hint Data/D5/depthMaps/depthLFT_3379.tif --depth Data/D5/depthMaps/depthLFT_3379.tif --prefix 3379_hybrid_
_
python graph.py --original Data/D5/Raw/LFT_3379.NEF --mode Hybrid --depth Data/D5/depthMaps/depthLFT_3379.tif --prefix 3379_
