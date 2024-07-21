import OpenEXR

from kripto_utils import Get_Window_Shape, Get_Cryptomattes_From_Header, identify_channels, parse_manifest
from kripto_logger import Setup_Logger

logger = Setup_Logger()
exr_file_path = "sample.exr"
logger.info("Reading Sample Exr file")
sample_exr = OpenEXR.InputFile(exr_file_path)

logger.info("Reading Sample Exr file headers")
header = sample_exr.header()

dataWindow_shape = Get_Window_Shape('dataWindow', header)
displayWindow_shape = Get_Window_Shape('displayWindow', header)

if dataWindow_shape != displayWindow_shape:
    logger.warning("The size of dataWindow and displayWindow do not match.")
    logger.warning("This may be normal behaviour for EXR Image, however,"
                   "the author of this script does not expect such behaviour.")
    logger.warning("On case of conflict, dataWindow size will be used.")

logger.info(f"Amount of pixels is {dataWindow_shape[0] * dataWindow_shape[1]}")

"""
TODO:
1. Get all Object, Material and Asset Names
2. Compute Float Point ID for all Names
3. Get all Cryptomatte layer pairs
    3.1 Half or Full?
    3.2 Coverage?
4. Extra Masks for all names from given layer pairs
5. Assign different color to layer
6. Export Separate colored layers and combined layers for Object, Material and Asset.
"""

cryptomattes = Get_Cryptomattes_From_Header(header)
logger.info(f"Cryptomattes loading finished.")
assert len(cryptomattes) > 0, "There is no Cryptomattes available."

logger.info(f"Checking all mattes have [name]...")
for metadata_id, value in cryptomattes.items():
    if not "name" in value:
        logger.info(f"It seems that cryptomatte {metadata_id} has no name, assigning empty name.")
        cryptomattes[metadata_id]["name"] = ""

default_cryptomattes_id = sorted(list(cryptomattes.keys()), key=lambda x: cryptomattes[x]["name"])[0]
all_cryptomattes_id = sorted(list(cryptomattes.keys()), key=lambda x: cryptomattes[x]["name"])

logger.info("Looking for cryptomatte related channels in Cryptomatte.")
for metadata_id, value in cryptomattes.items():
    name = value["name"]
    logger.info(f"Getting Channel names for {name.decode('utf-8')}")
    channels = identify_channels(header, name)
    cryptomattes[metadata_id]["channels"] = channels
    logger.info(f"Got channels: {channels}")

for each_cryptomatte_id in all_cryptomattes_id:
    logger.info(
        f"Computing float values from names(trimmed hash) in {cryptomattes[each_cryptomatte_id]['name'].decode('utf-8')}")
    cryptomattes[each_cryptomatte_id]['name2ID'], cryptomattes[each_cryptomatte_id]['ID2name'] = parse_manifest(
        cryptomattes, each_cryptomatte_id, exr_file_path)
