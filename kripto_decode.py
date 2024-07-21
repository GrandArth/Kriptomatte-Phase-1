import os.path
import OpenEXR
from PIL import Image
import numpy as np
import logging
from kripto_utils import Get_Window_Shape, Get_Cryptomattes_From_Header, identify_channels, parse_manifest, \
    get_masks_for_all_objs, id_to_rgb, get_combined_mask, apply_random_colormap_to_mask
import kripto_logger
from kripto_color import RandomColor


logger = logging.getLogger(__name__)

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
for metadata_id, metadata_dictionary in cryptomattes.items():
    if "name" not in metadata_dictionary:
        logger.info(f"It seems that cryptomatte {metadata_id} has no name, assigning empty name.")
        cryptomattes[metadata_id]["name"] = ""

# all_cryptomattes_id = sorted(list(cryptomattes.keys()), key=lambda x: cryptomattes[x]["name"])

logger.info("Looking for cryptomatte related channels in Cryptomatte.")
logger.info("Updating relevant attributes....")


def Create_Folder_for_Crypto_layer(var_exr_file_path, var_crypto_layer_name):
    current_file_dirname = os.path.dirname(var_exr_file_path)
    current_crypto_layer_folder = os.path.join(current_file_dirname, var_crypto_layer_name.decode('utf-8'))
    if not os.path.exists(current_crypto_layer_folder):
        os.mkdir(current_crypto_layer_folder)
    return current_crypto_layer_folder


for metadata_id, metadata_dictionary in cryptomattes.items():
    crypto_layer_name = metadata_dictionary["name"]

    crypto_layer_folder = Create_Folder_for_Crypto_layer(exr_file_path, crypto_layer_name)

    logger.info(f"Getting Channel names for {crypto_layer_name.decode('utf-8')}")
    channels, naming_scheme = identify_channels(header, crypto_layer_name)
    cryptomattes[metadata_id]["channels"] = channels
    cryptomattes[metadata_id]["naming_scheme"] = naming_scheme
    logger.info(f"Got channels: {channels}")

    logger.info(f"Computing float values from names(trimmed hash) in"
                f" {cryptomattes[metadata_id]['name'].decode('utf-8')}")
    (cryptomattes[metadata_id]['manifest'],
     cryptomattes[metadata_id]['name2ID'],
     cryptomattes[metadata_id]['ID2name']) = parse_manifest(cryptomattes,
                                                            metadata_id,
                                                            exr_file_path)
    logger.info(f"Dictionary format manifest created.")
    logger.info(f"Float values (ID) bi-directional lookup table created.")

    logger.info(f"Object mask for {crypto_layer_name}, will be wrote in {crypto_layer_folder}")
    for object_name, object_crypto_mask in get_masks_for_all_objs(var_exr=sample_exr, var_cryptomattes=cryptomattes,
                                                                  var_header=header,
                                                                  var_crypto_matte_id=metadata_id,
                                                                  var_window_shape=dataWindow_shape):
        if object_crypto_mask.min() == object_crypto_mask.max():
            logger.info(f"{crypto_layer_name}/{object_name} is not in current frame, not writing masks.")
            logger.info(f"It would be empty anyway.")
        else:
            logger.info(f"Writing {object_name} mask.")
            object_crypto_mask = np.repeat(np.expand_dims(object_crypto_mask, axis=2), 4, axis=2)

            object_id = cryptomattes[metadata_id]['name2ID'][object_name]
            preview_color = id_to_rgb(object_id)

            # Convert to RGBA format
            crypto_mask_image = Image.fromarray(object_crypto_mask, mode='RGBA')
            crypto_mask_image.save(os.path.join(f"{crypto_layer_folder}", f"{object_name}_mask.png"))

    logger.info("Computing combined mask.")
    combined_mask, name_to_mask_id_map = get_combined_mask(var_exr=sample_exr, var_cryptomattes=cryptomattes,
                                                           var_header=header, var_crypto_matte_id=metadata_id,
                                                           var_window_shape=dataWindow_shape)
    logger.info(f"Assigning Random Color to Object")
    random_color_generator = RandomColor(0, 0, 0.07)
    colored_combined_mask = apply_random_colormap_to_mask(mask_combined=combined_mask,
                                                          var_random_color=random_color_generator)
    crypto_mask_image = Image.fromarray(colored_combined_mask, mode='RGB')
    logger.info(f"Writing Combined Mask to {os.path.dirname(exr_file_path)}"
                f" / {crypto_layer_name.decode('utf-8')}_mask.png")
    crypto_mask_image.save(os.path.join(f"{os.path.dirname(exr_file_path)}",
                                        f"{crypto_layer_name.decode('utf-8')}_mask.png"))
