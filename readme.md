![Kriptomatte_icon.png](Kriptomatte_icon.png)
# Kriptomatte: Making Cryptomattes Easier for 2D Artists
This document outlines the Kriptomatte project, which aims to bring Cryptomatte functionality to Krita, the popular open-source painting software.

[中文](readme_sc.md)

## What is Kriptomatte and What is this repo?

Kriptomatte is a tool that helps artists work with Cryptomattes, which are special masks embedded within EXR image files. These masks isolate objects, materials, and assets, making them easier to select and manipulate in video post-processing software. 

Unfortunately, typical 2D image editing software doesn't work well with Cryptomattes, so artists often rely on ID Passes. While many popular software packages, such as Blender, lack the option to render ID passes, the codes in this repo (Kriptomatte Phase 1) offer a solution. Kriptomatte converts Cryptomatte information into ID Pass-like masks that can be used in painting software like Krita, Photoshop, and Clip Studio.

![Kriptomatte Sample.png](Kriptomatte%20Sample.png)

Seperated Masks are also exported for a finer control.

![Kriptomatte_Seperate_Mask.png](Kriptomatte_Seperate_Mask.png)

## Project Phases

The Kriptomatte project is planned for four phases:

### Phase 1: Python Script (Current Phase)

https://pypi.org/project/kriptomatte/

This phase focuses on creating a Python script that can extract Cryptomatte information from EXR files and convert it into separate, colored PNG images with alpha channels. This makes Cryptomattes more accessible to artists who can use these PNGs for painting in any software they prefer.

### Phase 2: Standalone Executable

Phase 2 will rewrite the Python script in C++ to create a faster, standalone executable program. This program won't require any command-line experience; artists can simply drag and drop their EXR files onto the executable to generate the PNGs.

### Phase 3a & 3b: Krita Integration

The final phases involve integrating Kriptomatte functionality directly into Krita. Phase 3a will focus on creating a C++ port of the code, while Phase 3b will develop a Python plugin for Krita.


# Usage

## Installation 

You can install the script using `pip` from your **command prompt** (also known as a **terminal**). If you're unfamiliar with the concept, think of it as a special window where you type commands for your computer to execute. Here's how to open it on Windows: **hold Shift** and **right-click** on your desktop, then select **"Open PowerShell here"** or **"Open Windows Terminal here."**  Once the terminal opens, copy and paste one of the following commands to install:

```bash
python -m pip install kriptomatte
# or
pip install kriptomatte
```

## Use from Terminal

After installation, you can use the script in your terminal. Open the terminal again, navigate to the directory containing your EXR file, and run the following command, replacing "path/to/your/file.exr" with the actual path to your file:

To easily find the path to your EXR file on Windows, left-click it, then type `alt + 3`.
This will copy the file location to your clipboard. You can then paste it into the command.
If the hotkey doesn't work, you can find a light blue option to copy the file path in the top-left corner of your File Explorer.

```bash
kriptomatte -i "path to your exr file"
# for example
kriptomatte  -i "C:\Users\GrandArth\Pictures\sample.exr"
```

## Important Note:

Currently, the script only supports Cryptomattes stored in 32-bit EXR files. Ensure your EXR files are rendered with 32-bit precision for the script to work correctly.

# Code Reference

```ref
Friedman, Jonah, and Andrew C. Jones. 2015. “Fully Automatic ID Mattes with Support for Motion Blur and Transparency.” In ACM SIGGRAPH 2015 Posters, 1–1. Los Angeles California: ACM. https://doi.org/10.1145/2787626.2787629.
“OpenEXR Bindings for Python.” n.d. Accessed July 17, 2024. https://www.excamera.com/sphinx/articles-openexr.html.
“Psyop/Cryptomatte.” (2015) 2024. Python. Psyop. https://github.com/Psyop/Cryptomatte.
“Synthesis-AI-Dev/Exr-Info: Package with Helper Modules to Process EXR Files Generated by Renders.” n.d. Accessed July 21, 2024. https://github.com/Synthesis-AI-Dev/exr-info.
```