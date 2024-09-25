from setuptools import setup, find_packages
import subprocess
import os

setup(
    name='LookBuilderPipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # other dependencies
    ],
    cmdclass={
        'develop': type('CustomDevelopCommand', (setuptools.command.develop.develop,), {'run': lambda self: clone_repo() or setuptools.command.develop.develop.run(self)}),
        'install': type('CustomInstallCommand', (setuptools.command.install.install,), {'run': lambda self: clone_repo() or setuptools.command.install.install.run(self)}),
    },
)
