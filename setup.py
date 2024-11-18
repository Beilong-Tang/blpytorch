from setuptools import setup, find_packages

setup(
    name="blpytorch",                # Replace with your package name
    version="0.1.0",                  # Version number
    author="Beilong Tang",               # Your name
    author_email="bt132@duke.edu",
    description="My pytorch package",  # Short description
    long_description=open("README.md").read(),  # Optional: Read from README
    long_description_content_type="text/markdown",
    url="https://github.com/Beilong-Tang/blpytorch",  # Optional: Project URL
    packages=find_packages(),           # Automatically find sub-packages
    classifiers=[                       # Optional: Metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',            # Python version requirement
    install_requires=[                  # Add dependencies
        # Example: "numpy>=1.21.0",
    ],
)