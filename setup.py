from setuptools import setup

setup(
    name='Optimized_OCR',
    packages=['trainer'],
    include_package_data=True,
    install_requires=[
        'flask',
        'tensorflow',
        'cv2',
        'numpy',
        'werkzeug',
        'six',
        'scikit-learn'
    ],
)
