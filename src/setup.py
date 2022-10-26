# これから作るパッケージの依存情報、バージョン情報、パッケージ名を設定するファイル

from setuptools import setup, find_packages

setup(
    name='my_utils',
    description='Python Distribution Utilities',
    install_requires=['pandas', 'scikit-learn'],
    packages=find_packages()
)
