# intuitive-arm-reach
ECE496: Capstone Project

# Create Project Environment (e.g. with Conda or Venv)

### Venv example shown below
`cd my-project`

`python -m venv arm_reach`

`source arm_reach/bin/activate`

# Installing dependencies

Most project dependencies can be installed through the requirements.txt file

`pip3 install -r requirements.txt`

Robotics Toolbox requires git cloning Peter Corke's Repo to get the bleeding edge version. Run the following script to clone and/or pull the latest version of the repo and load it as a python module:

`./setup.sh`

If you're getting a permission error try:

`chmod +x setup.sh`

# Proximodistal Freezing-Freeing Exploration Demo
This algorithm is the primary method of encoding and learning new motions in our project. You can find a demo [here](https://pranshumalik14.github.io/intuitive-arm-reach/notebooks/pdff_impl.jl.html). More details on how it has been used can be found in our poster and project report.
