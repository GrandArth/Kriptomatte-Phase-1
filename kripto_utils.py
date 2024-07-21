import json
import os
import re
import struct
import colorsys
import numpy as np
import random
import pymmh3 as mmh3
from kripto_datatypes import ExrDtype, pixel_dtype, numpy_dtype
from kripto_logger import Setup_Logger

CRYPTO_METADATA_LEGAL_PREFIX = ["exr/cryptomatte/", "cryptomatte/"]
CRYPTO_METADATA_DEFAULT_PREFIX = CRYPTO_METADATA_LEGAL_PREFIX[1]

root_logger = Setup_Logger()


def Get_Window_Shape(window_type: str, input_header):
    root_logger.info(f"Reading {window_type} Size Information")
    current_window = input_header[window_type]
    current_window_width = current_window.max.x - current_window.min.x + 1
    current_window_height = current_window.max.y - current_window.min.y + 1
    current_window_shape = (current_window_width, current_window_height)
    root_logger.info(f"The size of {window_type} is {current_window_width}x{current_window_height}")
    return current_window_shape


def Get_Cryptomattes_From_Header(exr_header):
    temp_cryptomattes = {}
    root_logger.info("Getting all cryptomattes")
    for key, value in exr_header.items():
        for prefix in CRYPTO_METADATA_LEGAL_PREFIX:
            if not key.startswith(prefix):
                continue
            numbered_key = key[len(prefix):]
            root_logger.info(f"Trimming prefix {key} to {numbered_key}")
            metadata_id, partial_key = numbered_key.split("/")
            root_logger.info(f"Split {numbered_key} to {metadata_id} and {partial_key}")
            if metadata_id not in temp_cryptomattes:
                root_logger.info(f"Adding key: {metadata_id} to cryptomattes.")
                temp_cryptomattes[metadata_id] = {}
            temp_cryptomattes[metadata_id][partial_key] = value
            root_logger.info(f"Assigning {type(value)} value of length {len(value)} to {metadata_id}/{partial_key}")
            temp_cryptomattes[metadata_id]['md_prefix'] = prefix
            break
    return temp_cryptomattes


def identify_channels(input_header, input_name):
    """from a name like "cryptoObject",
    gets sorted channels, such as cryptoObject00, cryptoObject01, cryptoObject02
    """
    input_name = input_name.decode('utf-8')
    channel_list = list(input_header['channels'].keys())
    # regex for "cryptoObject" + digits + ending with .red or .r
    channel_regex = re.compile(r'({name}\d+)\.(?:red|r|R)$'.format(name=input_name))
    pure_channels = []
    for channel in channel_list:
        match = channel_regex.match(channel)
        if match:
            pure_channels.append(match.group(1))
    return sorted(pure_channels)


def resolve_manifest_paths(exr_path, sidecar_path):
    if "\\" in sidecar_path:
        print("Cryptomatte: Invalid sidecar path (Back-slashes not allowed): ", sidecar_path)
        return ""  # to enforce the specification.
    joined = os.path.join(os.path.dirname(exr_path), sidecar_path)
    return os.path.normpath(joined)


def Load_inFile_Manifest(input_cryptomattes, selection):
    try:
        manifest_bytes = input_cryptomattes[selection]['manifest']
    except KeyError:
        root_logger.warning(f'Input Cryptomattes dose not have {selection}?')
        root_logger.warning(f'Cryptomattes {selection} dose not have keyword [manifest]?')
        raise KeyError
    manifest_string = manifest_bytes.decode('utf-8')
    # Convert JSON string to dictionary
    manifest_dict = json.loads(manifest_string)
    return manifest_dict


def parse_manifest(input_cryptomattes, selection, var_exr_file_path):
    """ Loads json manifest and unpacks hex strings into floats,
    and converts it to two dictionaries, which map IDs to names and vice versa.
    Also caches the last manifest in a global variable so that a session of selecting
    things does not constantly require reloading the manifest (' ~0.13 seconds for a
    32,000 name manifest.')
    """
    num = selection
    manifest = {}

    manif_file = input_cryptomattes[num].get("manif_file", False)  # If no manif_file return false
    if manif_file:
        root_logger.info("Side car manifest detected in EXR file, loading external file.")
        manif_file = manif_file.decode('utf-8')
        root_logger.info(f"External manif_file name {manif_file}")
        manif_file = resolve_manifest_paths(var_exr_file_path, manif_file)
        root_logger.info(f"Full EXR file path is {manif_file}")

    if manif_file:
        if os.path.exists(manif_file):
            with open(manif_file) as json_data:
                manifest = json.load(json_data)
        else:
            print("Cryptomatte: Unable to find manifest file: ", manif_file)
    else:
        manifest = Load_inFile_Manifest(input_cryptomattes, selection)

    from_names = {}
    from_ids = {}

    unpacker = struct.Struct('=f')
    packer = struct.Struct("=I")
    for name, value in manifest.items():
        packed = packer.pack(int(value, 16))
        packed = packed = b'\0' * (4 - len(packed)) + packed
        id_float = unpacker.unpack(packed)[0]
        name_str = name if type(name) is str else name.encode("utf-8")
        from_names[name_str] = id_float
        from_ids[id_float] = name_str

    input_cryptomattes[num]["names_to_IDs"] = from_names
    input_cryptomattes[num]["ids_to_names"] = from_ids

    return from_names, from_ids


def mm3hash_float(input_name):
    hash_32 = mmh3.hash(input_name)
    exp = hash_32 >> 23 & 255
    if (exp == 0) or (exp == 255):
        hash_32 ^= 1 << 23

    packed = struct.pack('<L', hash_32 & 0xffffffff)
    return struct.unpack('<f', packed)[0]


def name_to_ID(input_name):
    return mm3hash_float(input_name)


def get_channel_precision(input_header, channel_name: str):
    """Get the precision of a channel within the EXR"""
    if channel_name not in input_header['channels']:
        raise TypeError(
            f"There is no channel called '{channel_name}' in EXR file. "
        )
    # Use values in dict to find matching key (values are unique)
    for exr_d, pix_d in pixel_dtype.items():
        prec = input_header["channels"][channel_name].type
        if prec == pix_d:
            return exr_d


def read_channel(input_exr_file, input_header, channel_name: str, exr_window_shape,
                 cast_dtype: [ExrDtype] = None) -> np.ndarray:
    chan_dtype = get_channel_precision(input_header, channel_name)
    np_type = numpy_dtype[chan_dtype]
    channel_arr = np.frombuffer(input_exr_file.channel(channel_name), dtype=np_type)
    channel_arr = channel_arr.reshape(exr_window_shape)
    channel_arr = channel_arr.copy()  # Arrays read from buffers can be read-only

    if cast_dtype is not None:
        if not isinstance(cast_dtype, ExrDtype):
            raise ValueError(f"Expected type {ExrDtype.__name__}. Got: {type(cast_dtype)}")
        channel_arr = channel_arr.astype(numpy_dtype[cast_dtype])

    return channel_arr


def read_all_channels_in_list(input_exr_file, channel_list,
                              input_header, exr_window_shape,
                              cast_dtype: [ExrDtype] = None):
    list_of_read_channels = []
    for each_channel in channel_list:
        list_of_read_channels.append(read_channel(input_exr_file, input_header,
                                                  each_channel,
                                                  exr_window_shape,
                                                  cast_dtype))
    return list_of_read_channels


def get_coverage_for_rank(float_id: float, cr_combined: np.ndarray, rank: int) -> np.ndarray:
    """
    Get the coverage mask for a given rank from cryptomatte layers

    Args:
        float_id (float32): The ID of the object
        cr_combined (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                     By default, there are 3 layers, corresponding to a level of 6.
        rank (int): The rank, or level, of the coverage to be calculated

    Returns:
        numpy.ndarray: Mask for given coverage rank. Dtype: np.float32, Range: [0, 1]
    """
    id_rank = cr_combined[:, :, rank * 2] == float_id
    coverage_rank = cr_combined[:, :, rank * 2 + 1] * id_rank

    return coverage_rank


def get_mask_for_id(obj_hex_id: str, channels_arr: np.ndarray, level: int = 6) -> np.ndarray:
    """
    Extract mask corresponding to a float id from the cryptomatte layers

    Args:
        obj_hex_id (str): The ID of the object (from manifest).
        channels_arr (numpy.ndarray): The cryptomatte layers combined into a single array along the channels axis.
                                     Each layer should be in acsending order with it's channels in RGBA order.
                                     By default, there are 3 layers, corresponding to a level of 6.
        level (int): The Level of the Cryptomatte. Default is 6 for most rendering engines. The level dictates the
                     max num of objects that the crytomatte can represent. The number of cryptomatte layers in EXR
                     will change depending on level.

    Returns:
        numpy.ndarray: Mask from cryptomatte for a given id. Dtype: np.uint8, Range: [0, 255]
    """
    float_id = name_to_ID(obj_hex_id)

    coverage_list = []
    for rank in range(level):
        coverage_rank = get_coverage_for_rank(float_id, channels_arr, rank)
        coverage_list.append(coverage_rank)
    coverage = sum(coverage_list)
    coverage = np.clip(coverage, 0.0, 1.0)
    mask = (coverage * 255).astype(np.uint8)
    return mask


def Get_Names_for_ALl_Sub_Crypto_channel(crypto_channels, name_type="UpSingle"):
    list_of_sub_crypto_channels = []
    if name_type == "UpSingle":
        for each_channel_prefix in crypto_channels:
            crypto_channel_RGBA_dict = {
                "R":f"{each_channel_prefix}.R",
                "G": f"{each_channel_prefix}.G",
                "B": f"{each_channel_prefix}.B",
                "A": f"{each_channel_prefix}.A"
            }
            list_of_sub_crypto_channels.append(crypto_channel_RGBA_dict)
    elif name_type == "SmallSingle":
        for each_channel_prefix in crypto_channels:
            crypto_channel_RGBA_dict = {
                "R": f"{each_channel_prefix}.r",
                "G": f"{each_channel_prefix}.g",
                "B": f"{each_channel_prefix}.b",
                "A": f"{each_channel_prefix}.a",
            }
            list_of_sub_crypto_channels.append(crypto_channel_RGBA_dict)
    elif name_type == "Full":
        for each_channel_prefix in crypto_channels:
            crypto_channel_RGBA_dict = {
                "R": f"{each_channel_prefix}.red",
                "G": f"{each_channel_prefix}.green",
                "B": f"{each_channel_prefix}.blue",
                "A": f"{each_channel_prefix}.alpha",
            }
            list_of_sub_crypto_channels.append(crypto_channel_RGBA_dict)
    else:
        root_logger.warning(f"No crypto channel naming scheme (A|a|alpha?) provide, default to (A)")
        for each_channel_prefix in crypto_channels:
            crypto_channel_RGBA_dict = {
                "R":f"{each_channel_prefix}.R",
                "G": f"{each_channel_prefix}.G",
                "B": f"{each_channel_prefix}.B",
                "A": f"{each_channel_prefix}.A"
            }
            list_of_sub_crypto_channels.append(crypto_channel_RGBA_dict)
    return list_of_sub_crypto_channels




def get_masks_for_all_objs(var_exr, var_cryptomattes, var_header, var_crypto_matte_id: str, var_window_shape):
    """
    Get an individual mask of every object in the cryptomatte

    Args:
        var_crypto_matte_id: The name of the cryptomatte definition from which to extract masks

    Returns:
        collections.OrderedDict(str, numpy.ndarray): Mapping from the name of each object to it's anti-aliased mask.
            For mask -> Shape: [H, W], dtype: uint8
    """
    try:
        manifest = var_cryptomattes[var_crypto_matte_id]['name2ID']
    except KeyError:
        raise KeyError(f"name2ID not found in cryptomatte {var_crypto_matte_id}.")

    crypto_channels = var_cryptomattes[var_crypto_matte_id]['channels']
    crypto_channels_rgba = Get_Names_for_ALl_Sub_Crypto_channel(crypto_channels, "UpSingle")
    channels_arr = np.stack(read_all_channels_in_list(input_exr_file=var_exr,
                                                      channel_list=crypto_channels_rgba,
                                                      input_header=var_header,
                                                      exr_window_shape=var_window_shape), axis=-1)

    # Number of layers depends on level of cryptomatte: ``num_layers = math.ceil(level / 2)``. Default level = 6.
    # Each layer has 4 channels: RGBA
    num_layers, num_level = Get_Quantity_of_Cryptolayers(crypto_channels)

    # The objects in manifest are sorted alphabetically to maintain some order.
    # Each obj is assigned a unique ID (per image) for the mask
    obj_names = sorted(manifest.keys())
    for obj_name in obj_names:
        obj_hex_id = manifest[obj_name]
        mask = get_mask_for_id(obj_hex_id, channels_arr, num_level)
        yield obj_name, mask


def Get_Quantity_of_Cryptolayers(crypto_channels):
    num_layers = len(crypto_channels)
    num_level = 2 * num_layers
    return num_layers, num_level


def get_combined_mask(var_crypto_matte_id: str):
    """
    Get a single mask for semantic segmentation representing all the objects within the scene.
    Each object is represented by a unique integer value, starting from 1. 0 is reserved for background.

    Args:
        var_crypto_matte_id: The name of the cryptomatte definition from which to extract masks

    Returns:
        numpy.ndarray: Mask of all objects. Shape: [H, W], dtype: np.uint16.
        dict: Mapping of the object names to mask IDs for this image.
    """
    obj_masks = get_masks_for_all_objs(var_crypto_matte_id)

    best = None
    total = None
    mask_combined = None
    name_to_mask_id_map = {"background": 0}

    for (idx, (obj_name, obj_mask)) in enumerate(obj_masks):
        name_to_mask_id_map[obj_name] = idx + 1
        if mask_combined is None:
            mask_combined = np.zeros_like(obj_mask, dtype=np.uint16)
        if best is None:
            best = np.zeros_like(obj_mask)
        if total is None:
            total = np.zeros_like(obj_mask)

        total += obj_mask
        mask_combined[obj_mask > best] = idx + 1
        best = np.max([best, obj_mask], axis=0)

    mask_combined[255 - total > best] = 0

    return mask_combined, name_to_mask_id_map


def random_color():
    hue = random.random()
    sat, val = 0.7, 0.7
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    rgb = []
    for col in [r, g, b]:
        col_np = np.array(col, dtype=np.float32)
        col_np = (np.clip(col_np * 255, 0, 255)).astype(np.uint8)
        col_list = col_np.tolist()
        rgb.append(col_list)
    return rgb


def apply_random_colormap_to_mask(mask_combined: np.ndarray) -> np.ndarray:
    """
    Apply random colors to each segment in the mask, for visualization
    """
    num_objects = mask_combined.max() + 1
    colors = [[0, 0, 0]] + [random_color() for _ in range(num_objects - 1)]  # Background is fixed color: black
    mask_combined_rgb = np.take(colors, mask_combined, 0)
    return mask_combined_rgb.astype(np.uint8)
