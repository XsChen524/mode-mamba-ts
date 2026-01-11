# Parallel experiments across GPUs

# # ETT experiments
# bash ./scripts/multivariate_forecasting/ETT/MODE_ETTh1.sh &
# bash ./scripts/multivariate_forecasting/ETT/MODE_ETTh2.sh &
# bash ./scripts/multivariate_forecasting/ETT/MODE_ETTm1.sh &
# bash ./scripts/multivariate_forecasting/ETT/MODE_ETTm2.sh &
# wait

# # ECL + Exchange
# bash ./scripts/multivariate_forecasting/ECL/MODE.sh &
bash ./scripts/multivariate_forecasting/Exchange/MODE.sh &

# # Weather + Solar
# bash ./scripts/multivariate_forecasting/Weather/MODE.sh &
# bash ./scripts/multivariate_forecasting/SolarEnergy/MODE.sh &

# # Traffic + PEMS
# bash ./scripts/multivariate_forecasting/Traffic/MODE.sh &
# bash ./scripts/multivariate_forecasting/PEMS/MODE_03.sh &
# bash ./scripts/multivariate_forecasting/PEMS/MODE_07.sh &
# bash ./scripts/multivariate_forecasting/PEMS/MODE_04.sh &
# bash ./scripts/multivariate_forecasting/PEMS/MODE_08.sh &
wait

echo "All experiments completed!"
