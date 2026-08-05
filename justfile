set dotenv-load	:= true #load the .env file for folder paths

# Run the full pipeline
run-pipeline: copy-data
    # Run the Python file
    uv run snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime --rerun-incomplete -q rules --rerun-triggers mtime

# Copy raw data into session folders
copy-data:
    uv run python workflow/scripts/00_create_session_folders.py

# search for the pattern in the data folders
find-session pattern:
  fd --type d {{pattern}} $SESSION_ROOT_DIR

find-data pattern:
  fd -e ppd -e tsv {{pattern}} $SESSION_ROOT_DIR 



# Search for and execute workflow in a session folder
make-session SESSION_ID *FLAGS:
  #!/usr/bin/bash 
  #shebang is necessary otherwise it will be executed line by line indepedently
  target=$(fd --type d --full-path '{{SESSION_ID}}$' $SESSION_ROOT_DIR)
  echo "$target" | while read line; do echo "$line"; done
  read -p "Are you sure you want to proceed? [y/N] " ans; \
  if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
      echo "Aborted."; exit 1; \
  fi
  targets=$(echo "$target" | awk '{printf "%sprocessed/pycontrol_workflow.done ", $0}')
  echo $targets
  uv run snakemake $targets --snakefile workflow/pycontrol.smk -c20 -k {{FLAGS}} -q rules --rerun-incomplete --rerun-triggers mtime


# Search for and execute workflow in a session folder
make SEARCH_TERM *FLAGS:
  #!/usr/bin/bash 
  #shebang is necessary otherwise it will be executed line by line indepedently
  target=$(fd --type d --full-path '{{SEARCH_TERM}}.*-[0-9]{6}$' $SESSION_ROOT_DIR)
  echo "$target" | while read line; do echo "$line"; done
  read -p "Are you sure you want to proceed? [y/N] " ans; \
  if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
      echo "Aborted."; exit 1; \
  fi
  targets=$(echo "$target" | awk '{printf "%sprocessed/pycontrol_workflow.done ", $0}')
  echo $targets
  uv run snakemake $targets --snakefile workflow/pycontrol.smk -c20 -k {{FLAGS}} -q rules --rerun-incomplete --rerun-triggers mtime

#Search for and execute the sorting workflow in a session folder
sort SEARCH_TERM *FLAGS:
    #!/usr/bin/bash 
    #shebang is necessary otherwise it will be executed line by line indepedently
    target=$(fd --type d --full-path '{{SEARCH_TERM}}.*-[0-9]{6}$' $SESSION_ROOT_DIR)
    echo "$target" | while read line; do echo "$line"; done
    read -p "Are you sure you want to proceed? [y/N] " ans; \
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
        echo "Aborted."; exit 1; \
    fi
    targets=$(echo "$target" | awk '{printf "%sprocessed/spike_workflow.done ", $0}')
    uv run snakemake $targets --snakefile workflow/spikesort.smk -c20 -k {{FLAGS}} --rerun-incomplete -q rules --rerun-triggers mtime

#Search for and execute the sorting workflow in a session folder
sort-session SESSION_ID *FLAGS:
    #!/usr/bin/bash 
    #shebang is necessary otherwise it will be executed line by line indepedently
    target=$(fd --type d --full-path '{{SESSION_ID}}$' $SESSION_ROOT_DIR)
    echo "$target" | while read line; do echo "$line"; done
    read -p "Are you sure you want to proceed? [y/N] " ans; \
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
        echo "Aborted."; exit 1; \
    fi
    targets=$(echo "$target" | awk '{printf "%sprocessed/spike_workflow.done ", $0}')
    uv run snakemake $targets --snakefile workflow/spikesort.smk -c20 -k {{FLAGS}} --rerun-incomplete -q rules --rerun-triggers mtime


# Search for and execute workflow in a session folder
run-model SEARCH_TERM *FLAGS:
  #!/usr/bin/bash 
  #shebang is necessary otherwise it will be executed line by line indepedently
  target=$(fd --type d --full-path '{{SEARCH_TERM}}.*-[0-9]{6}$' $SESSION_ROOT_DIR)
  echo "$target" | while read line; do echo "$line"; done
  read -p "Are you sure you want to proceed? [y/N] " ans; \
  if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
      echo "Aborted."; exit 1; \
  fi
  targets=$(echo "$target" | awk '{printf "%sprocessed/modelling_workflow.done ", $0}')
  echo $targets
  uv run snakemake $targets --snakefile workflow/modelling.smk -c20 -k {{FLAGS}} -q rules --rerun-incomplete --rerun-triggers mtime

run-model-session SESSION_ID *FLAGS:
  #!/usr/bin/bash 
  #shebang is necessary otherwise it will be executed line by line indepedently
  target=$(fd --type d --full-path '{{SESSION_ID}}$' $SESSION_ROOT_DIR)
  echo "$target" | while read line; do echo "$line"; done
  read -p "Are you sure you want to proceed? [y/N] " ans; \
  if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
      echo "Aborted."; exit 1; \
  fi
  targets=$(echo "$target" | awk '{printf "%sprocessed/modelling_workflow.done ", $0}')
  echo $targets
  uv run snakemake $targets --snakefile workflow/modelling.smk -c20 -k {{FLAGS}} -q rules --rerun-incomplete --rerun-triggers mtime



run-aligner:
  uv run fastapi dev apps/atlasaligner/backend/main.py &
  npm --prefix apps/atlasaligner/frontend run dev &