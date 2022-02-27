# Clone bleeding edge robotics toolbox, pull latest version, install as python module
git clone https://github.com/petercorke/robotics-toolbox-python.git
cd robotics-toolbox-python
git pull --rebase
pip3 install -e .