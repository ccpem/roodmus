#!/usr/bin/bash

debug=$"True"
boundary_investigation=$"True"
overlap_investigation=$"True"
depth_investigation=$"True"

margins=( '_margin0' '_margin200' '_margin500')
particles=( '_margin0_300particles' '_margin0_400particles' '_margin0_500particles' )

particles_per_ugraph=200
particle_diameter=300
box_height=150
box_width=150

for n in ${margins[@]}; do
  log_filepath=$"DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$".log"
  image_globpath=$"DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$"/image*.mrc"
  picked_particle_file=$"relion_projects/DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$"/Extract/job006/particles.star"

  python parse_parakeet_progeny.py --debug $debug --log_filepath $log_filepath --image_globpath "$image_globpath" --picked_particle_file $picked_particle_file --boundary_investigation $boundary_investigation --overlap_investigation $overlap_investigation --depth_investigation $depth_investigation --particle_diameter $particle_diameter --box_width $box_width --box_height $box_height --particles_per_ugraph $particles_per_ugraph
  wait $!
  outdir=$"Study"$n$"/"
  mkdir $outdir
  mv *.p?? $outdir
  mv $'picked_particles.yaml' $outdir
  mv $'truth_particles.yaml' $outdir
done

for n in ${particles[@]}; do
  log_filepath=$"DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$".log"
  image_globpath=$"DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$"/image*.mrc"
  picked_particle_file=$"relion_projects/DESRES-Trajectory_sarscov2-13795965-no-water_10000_conf_2kparticles"$n$"/Extract/job006/particles.star"

  particles_per_ugraph=$(($particles_per_ugraph+100))

  python parse_parakeet_progeny.py --debug $debug --log_filepath $log_filepath --image_globpath "$image_globpath" --picked_particle_file $picked_particle_file --boundary_investigation $boundary_investigation --overlap_investigation $overlap_investigation --depth_investigation $depth_investigation --particle_diameter $particle_diameter --box_width $box_width --box_height $box_height --particles_per_ugraph $particles_per_ugraph
  wait $!
  outdir=$"Study"$n$"/"
  mkdir $outdir
  mv *.p?? $outdir
  mv $'picked_particles.yaml' $outdir
  mv $'truth_particles.yaml' $outdir
done
