#!/bin/bash

export PATH=~/julia-1.6.1/bin:$PATH

echo "Starting Spike extraction using the julia language"
cd $1

/home/mobs/julia-1.6.1/bin/julia spikeFilter_singleSpikeDataset.jl --projectDir=$1 $2 $3 $4 $5 $6 $7 $8 $9
