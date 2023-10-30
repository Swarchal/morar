from setuptools import setup


def read_requirements():
    with open("requirements.txt", "r") as f:
        return [i.strip() for i in f.readlines()]


setup(
    name="morar",
    version="0.2",
    url="http://github.com/swarchal/morar",
    description="Data management for CellProfiler",
    author="Scott Warchal",
    author_email="s.warchal@sms.ed.ac.uk",
    license="MIT",
    packages=["morar"],
    python_requires=">=3.10",
    tests_require="pytest",
    install_requires=read_requirements(),
    zip_safe=True,
)
