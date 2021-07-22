#!/bin/bash

export PATH=~/julia-1.5.3/bin:$PATH

echo "Starting Spike extraction using the julia language"
cd $1

julia spikeFilter_withStride.jl --projectDir=$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
