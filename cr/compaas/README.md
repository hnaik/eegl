```shell
docker run -it --rm \
    --volume "/data/results:/home/egr/egr/.output" \
    --volume "./local_runs:/input" \
    --env="INPUT_FILE=/input/dev.yml" egr-job-<tag>