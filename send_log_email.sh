#!/bin/bash

# Define variables
LOG_DIR=".snakemake/log"      # The directory containing the log files
EMAIL_SUBJECT="Pipeline log $(date)"
EMAIL_TO="teristam@gmail.com"
EMAIL_FROM="sender@jade.com"   # Some email systems require this

# Find the latest log file in the directory
LATEST_LOG_FILE=$(ls -t "$LOG_DIR"/*.log 2> /dev/null | head -n 1)
echo $LATEST_LOG_FILE

# Check if a log file was found
if [ -z "$LATEST_LOG_FILE" ]; then
  echo "No log files found in directory: $LOG_DIR"
  exit 1
fi

# Send the latest log file content as the email body
cat "$LATEST_LOG_FILE" | mail -s "$EMAIL_SUBJECT" -r "$EMAIL_FROM" "$EMAIL_TO"

# Check if the mail command was successful
if [ $? -eq 0 ]; then
  echo "Email sent successfully to $EMAIL_TO"
else
  echo "Failed to send email"
fi
