## Build and deploy

1. After you update your code, run `npm run build`
2. `sudo docker compose --env-file <env file> up` to deploy

# Pre-requisite
- The `SESSION_ROOT_DIR` environment variable should be set properly
- The Allen Brain Atlas should be located at `<SESSION_ROOT_DIR>/allenccf`. It should have the `annotation_volume_10um_by_index.npy` and `structure_tree_safe_2017.csv` file.

## Development
At the root of the project,
For backend, run `uv run fastapi dev apps/atlasaligner/backend/main.py`
For frontend, run ` npm --prefix apps/atlasaligner/frontend run dev`
To test the built React app is working, `npx serve dist`

## Running the program
run `just run-aligner`