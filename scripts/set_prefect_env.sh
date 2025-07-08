#!/bin/bash

HOST="192.168.191.194"
USER="prefect"
PASS="pZn!cT21x^dR"
PORT=9876
DB="orion"
host=$(hostname)

# prefect config set PREFECT_ORION_DATABASE_CONNECTION_URL="sqlite+aiosqlite:////home/hnaik/.prefect/orion-${host}.db"
# prefect config set PREFECT_ORION_DATABASE_CONNECTION_URL="postgresql+asyncpg://${USER}:${PASS}@${HOST}:${PORT}/${DB}"
# prefect config set PREFECT_ORION_DATABASE_CONNECTION_URL="postgresql+asyncpg://ayyitcbp:m3HIblq32ohHcPsf5q72EcdLDJhdNn4W@ziggy.db.elephantsql.com/ayyitcbp"
prefect config set PREFECT_LOGGING_LEVEL="ERROR"
prefect config set PREFECT_LOGGING_SERVER_LEVEL="ERROR"
# prefect config set PREFECT_API_URL="http://192.168.12.205:4200/api"
# prefect config unset PREFECT_API_URL
# prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${USER}:${PASS}@${HOST}:${PORT}/${DB}"

export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${USER}:${PASS}@${HOST}:${PORT}/${DB}"
export PREFECT_LOGGING_ROOT_LEVEL=INFO
export PREFECT_LOGGING_HANDLELRS_ORION_LEVEL=ERROR
export PREFECT_LOGGING_COLORS=True
# export PREFECT_LOGGING_FORMATTERS_STANDARD_FLOW_RUN_FMT="%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s"
export PREFECT_LOGGING_MARKUP=True
