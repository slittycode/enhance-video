#!/bin/bash
mkdir -p review_pack

echo "--- Static Grep Checks ---" > review_pack/static_checks.txt
echo -e "\nrg 'tqdm' -n ." >> review_pack/static_checks.txt
rg "tqdm" -n . >> review_pack/static_checks.txt || true

echo -e "\nrg 'from \.' -n cli.py upscale_video.py" >> review_pack/static_checks.txt
rg "from \." -n cli.py upscale_video.py >> review_pack/static_checks.txt || true

echo -e "\nrg 'parse_cli_overrides|resolve_model|analyze_video_type|temporal-filter|--model' -n cli.py upscale_video.py" >> review_pack/static_checks.txt
rg "parse_cli_overrides|resolve_model|analyze_video_type|temporal-filter|--model" -n cli.py upscale_video.py >> review_pack/static_checks.txt || true

echo "--- Precedence Matrix Test ---" > review_pack/cli_precedence_logs.txt
echo "1) --profile max_quality --temporal-filter none" >> review_pack/cli_precedence_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --profile max_quality --temporal-filter none >> review_pack/cli_precedence_logs.txt 2>&1
echo -e "\n2) --profile max_quality --model realesrgan-x4plus-anime" >> review_pack/cli_precedence_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --profile max_quality --model realesrgan-x4plus-anime >> review_pack/cli_precedence_logs.txt 2>&1
echo -e "\n3) --type animation --profile max_quality" >> review_pack/cli_precedence_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type animation --profile max_quality >> review_pack/cli_precedence_logs.txt 2>&1
echo -e "\n4) --type auto --profile max_quality --temporal-filter none" >> review_pack/cli_precedence_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type auto --profile max_quality --temporal-filter none >> review_pack/cli_precedence_logs.txt 2>&1

echo "--- Model Selection Correctness ---" > review_pack/model_selection_logs.txt
echo -e "\nreal-life scale 2:" >> review_pack/model_selection_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type real-life --scale 2 >> review_pack/model_selection_logs.txt 2>&1
echo -e "\nreal-life scale 4:" >> review_pack/model_selection_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type real-life --scale 4 >> review_pack/model_selection_logs.txt 2>&1
echo -e "\nanimation scale 2:" >> review_pack/model_selection_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type animation --scale 2 >> review_pack/model_selection_logs.txt 2>&1
echo -e "\nanimation scale 4:" >> review_pack/model_selection_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type animation --scale 4 >> review_pack/model_selection_logs.txt 2>&1

echo "--- Auto Type Behavior ---" > review_pack/auto_type_logs.txt
python upscale_video.py dummy.mp4 -o out.mp4 --plan-only --type auto >> review_pack/auto_type_logs.txt 2>&1

echo "--- Pytest Report ---" > review_pack/pytest_report.txt
PYTHONPATH=. pytest -q >> review_pack/pytest_report.txt 2>&1 || true
echo -e "\n--- Focus test ---" >> review_pack/pytest_report.txt
PYTHONPATH=. pytest -q tests/test_model_selection.py >> review_pack/pytest_report.txt 2>&1 || true
