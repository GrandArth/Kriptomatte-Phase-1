# Kriptomatte Phase 1
The goal of Kriptomatte project is to provide Krita the opensource painting software with the ability to deal wth Cryptomattes in EXR images, thus the name Kriptomatte.

## Planing

The Kriptomatte project will be carried in 4 separate phases.

### Phase 1

Develop a Python script that, given an EXR file, can output combined and separated cryptomattes (which  means ID masks for Objects, Materials and Assets) in common image format that supports Alpha channel, namely PNG.

In this phase, my aim is to make Cryptomattes more accessible for 2D artists in general. With a simple `pip install kriptomatte`, artists can decode properly colored PNGs from EXR render results for painting in their favorite software, whatever it is.

### Phase 2

After Phase 1 go through testing and finalizing, I will rewrite codes in Phase 1 in C++. So that a single drag and drop executable can be built. The executable will be doing the exact same thing, only faster. The tool will also be more accessible, as artists won't need to have any experience with `pip` or any `terminal` in general.

### Phase 3a and 3b

Port the C++ code to Krita.
Make a Python plugin for Krita.

Upon initial inspection, I believe after Phase 2, implementation in Krita should be simple (Push the results to layers with whatever code that dose the thing.). That being said, I have no experience with QT, so I plan to do these in the last Phase. Meanwhile, I do hope ppl have more experience with Python plugin can help bring the Phase 1 code to krita real soon.