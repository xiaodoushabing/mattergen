#!/bin/bash

# To set this file executable, run: `chmod +x run_permutations.sh`


# export MODEL_NAME=dft_mag_density
# export RESULTS_PATH="results/$MODEL_NAME/"  # Samples will be written to this directory, e.g., `results/dft_mag_density`

# # Generate conditional samples with a target magnetic density of 0.15
# mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --properties_to_condition_on="{'dft_mag_density': 0.15}" --diffusion_guidance_factor=2.0


# Define the base model name
MODEL_NAME="dft_mag_density"

# Define the base results path
BASE_RESULTS_PATH="./results/$MODEL_NAME"

# Define the batch size
BATCH_SIZE=64

# Define the property to condition on and its base value
CONDITION_PROPERTY='dft_mag_density'

# Define the permutations of magnetic density and guidance factor
mag_density=(0.5 1.0 2.0 3.0 4.0 5.0)
guidance_factor=(1.0 2.0 3.0 4.0 5.0)

num_mag_density=${#mag_density[@]}
num_guidance_factor=${#guidance_factor[@]}
total_permutations=$((num_mag_density * num_guidance_factor))
count=0

for md in "${mag_density[@]}"; do
  for gf in "${guidance_factor[@]}"; do
    # Construct the specific results path for this permutation
    RESULTS_PATH="${BASE_RESULTS_PATH}_${md}_${gf}"

    # Print the command that will be executed (for debugging)
    echo "Running: mattergen-generate ${RESULTS_PATH} --pretrained-name=${MODEL_NAME} --batch_size=${BATCH_SIZE} --properties_to_condition_on='{\"${CONDITION_PROPERTY}\": ${md}}' --diffusion_guidance_factor=${gf}"

    # Execute the mattergen-generate command
    mattergen-generate "${RESULTS_PATH}" \
      --pretrained-name="${MODEL_NAME}" \
      --batch_size="${BATCH_SIZE}" \
      --properties_to_condition_on="{\"${CONDITION_PROPERTY}\": ${md}}" \
      --diffusion_guidance_factor="${gf}"

    echo "Finished running for magnetic density: ${md}, guidance factor: ${gf}"
    
    count=$((count+1))
    echo "Completed ${count}/${total_permutations} permutations"
    echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  done
done

echo "All ${total_permutations} permutations have been run."