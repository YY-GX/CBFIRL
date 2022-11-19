#!/bin/zsh


w_cbf=(10000 7500 5000 2500)
entro_coeff=(0.001 0.01)
generator_iter=(10)
max_kl=(0.01 0.1)
ent_methods=("regularized")
seeds=(10)

w_cbf=(500 1000 2000 3000 40000)
entro_coeff=(0.001 0.01)
generator_iter=(10)
max_kl=(0.01 0.1)
ent_methods=("regularized")
seeds=(10)


w_cbf=(1000 2000)
entro_coeff=(0.01)
generator_iter=(10)
max_kl=(0.01 0.001 0.0)
ent_methods=("regularized")
seeds=(10)




#
#w_cbf=(10000 1000 100)
#entro_coeff=(0.001)
#generator_iter=(10)
#max_kl=(0.01)
#ent_methods=("regularized")
#
#w_cbf=(10000 7500)
#entro_coeff=(0.001)
#generator_iter=(10)
#max_kl=(0.01)
#ent_methods=("regularized")
#seeds=(1 2 3 4 5 6 7 8 9 10)


i=1
FOLDER="data/8obs/"


w_cbf=(1000)
entro_coeff=(0.01)
generator_iter=(10)
max_kl=(0.01)
ent_methods=("regularized")
seeds=(1 2 3 4 5 6 7 8 9 10)

cd .. && pwd &&
for w in "${w_cbf[@]}";
  do
    for entro in "${entro_coeff[@]}";
      do
        for num_gen in "${generator_iter[@]}";
          do
            for m_kl in "${max_kl[@]}";
              do
                for ent_m in "${ent_methods[@]}";
                  do
                    for s in "${seeds[@]}";
                      do
                        mainfolder="${FOLDER}cbf_airl_1_$i"
                        python scripts/train_trpo_cbf.py \
                         --seed $s \
                         --cbf_weight $w \
                         --epoch_num 30 \
                         --cbf_pth "data/8obs/pretrain_cbf/cbf"\
                         --restore_pth "data/8obs/pretrain_airl/airl"\
                         --is_restore 1 \
                         --is_freeze_discriminator 1 \
                         --is_auto_tuning 1 \
                         --generator_train_itrs $num_gen \
                         --policy_ent_coeff $entro \
                         --max_kl_step $m_kl \
                         --ent_method $ent_m \
                         --is_use_two_step 1 \
                         --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl" \
                         --demo_pth "src/demonstrations/8obs_acc_farther_target.pkl"
                        i=$(($i+1))
                      done
                  done
              done
          done
      done
  done





w_cbf=(1000)
entro_coeff=(0.01)
generator_iter=(10)
max_kl=(0.001)
ent_methods=("regularized")
seeds=(1 2 3 4 5 6 7 8 9 10)

pwd &&
for w in "${w_cbf[@]}";
  do
    for entro in "${entro_coeff[@]}";
      do
        for num_gen in "${generator_iter[@]}";
          do
            for m_kl in "${max_kl[@]}";
              do
                for ent_m in "${ent_methods[@]}";
                  do
                    for s in "${seeds[@]}";
                      do
                        mainfolder="${FOLDER}cbf_airl_2_$i"
                        python scripts/train_trpo_cbf.py \
                         --seed $s \
                         --cbf_weight $w \
                         --epoch_num 65 \
                         --cbf_pth "data/8obs/pretrain_cbf/cbf"\
                         --restore_pth "data/8obs/pretrain_airl/airl"\
                         --is_restore 1 \
                         --is_freeze_discriminator 1 \
                         --is_auto_tuning 1 \
                         --generator_train_itrs $num_gen \
                         --policy_ent_coeff $entro \
                         --max_kl_step $m_kl \
                         --ent_method $ent_m \
                         --is_use_two_step 1 \
                         --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl" \
                         --demo_pth "src/demonstrations/8obs_acc_farther_target.pkl"
                        i=$(($i+1))
                      done
                  done
              done
          done
      done
  done












#
## run based on just airl with diff seeds
#w_cbf=(7500)
#entro_coeff=(0.001)
#generator_iter=(10)
#max_kl=(0.01)
#ent_methods=("regularized")
#seeds=(1 2 3 4 5 6 7 8 9 10)
#
#
#i=1
#FOLDER="data/trpo_cbf/"
#PRETRAIN_FOLDER="data/just_airl/airl_"
#
#cd .. && pwd &&
#for w in "${w_cbf[@]}";
#  do
#    for entro in "${entro_coeff[@]}";
#      do
#        for num_gen in "${generator_iter[@]}";
#          do
#            for m_kl in "${max_kl[@]}";
#              do
#                for ent_m in "${ent_methods[@]}";
#                  do
#                    for s in "${seeds[@]}";
#                      do
#                        mainfolder="${FOLDER}cbf_airl_based_on_airl_with_diff_seeds_$i"
#                        python scripts/train_trpo_cbf.py \
#                         --seed $s \
#                         --cbf_weight $w \
#                         --epoch_num 75 \
#                         --cbf_pth "data/new_comb_new_demo/cbf_posx_posy/cbf"\
#                         --restore_pth ${PRETRAIN_FOLDER}$i"/airl"\
#                         --is_restore 1 \
#                         --is_freeze_discriminator 1 \
#                         --is_auto_tuning 1 \
#                         --generator_train_itrs $num_gen \
#                         --policy_ent_coeff $entro \
#                         --max_kl_step $m_kl \
#                         --ent_method $ent_m \
#                         --is_use_two_step 1 \
#                         --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
#                        i=$(($i+1))
#                      done
#                  done
#              done
#          done
#      done
#  done
#