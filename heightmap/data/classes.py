"""GAMUS land-cover classes.

From the paper (Xiong et al., GAMUS, arXiv:2305.14914):
    "annotated with six different land cover types, including
     1. ground; 2. low-vegetation; 3. building; 4. water; 5. road; 6. tree"

Ids are 1..6.  Id 0 appears in the rasters and is unlabelled/background -- rare
(~1% of pixels in the tiles inspected) -- so it is the default ignore_index.
Verify with scripts/dataset_stats.py before trusting this on the full release.
"""
UNLABELLED = 0
GROUND = 1
LOW_VEG = 2
BUILDING = 3
WATER = 4
ROAD = 5
TREE = 6

N_CLASSES = 7            # ids 0..6 inclusive
NAMES = {0: "unlabelled", 1: "ground", 2: "low-vegetation", 3: "building",
         4: "water", 5: "road", 6: "tree"}

#: classes an image-based shadow detector confuses with shadow, so they are excluded
#: from the shadow-consistency loss.  Dark tree canopy is the single largest source of
#: false positives; water and low vegetation are the next two.
SHADOW_EXCLUDE = (LOW_VEG, WATER, TREE)
