import setuptools
setuptools.setup(
    name='kriptomatte',
    version='0.0.1',
    scripts=['./kriptomatte'],
    author='Tempestas',
    entry_points = {'console_scripts': ['kritomatte=kritomatte.kripto_decode:cli']},
    description='Decode Cryptomattes in EXR file to PNG masks',
    packages=['lib.myscript'],
    install_requires=[
        'setuptools',
        'numpy',
        'pillow',
        'OpenEXR',
        'Imath'
    ],
    python_requires='>=3.5'
)