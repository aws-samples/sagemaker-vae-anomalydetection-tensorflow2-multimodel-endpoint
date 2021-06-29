#!/usr/bin/env bash
conda update -n base -c defaults conda -y;
conda env create -f environment.yml
