from scripts.utility import save_feature
from scripts.features import areole_area

save_feature(areole_area, skip_existing=True, clean=False)
